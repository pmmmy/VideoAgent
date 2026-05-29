# -*- coding: utf-8 -*-
"""
阶段5: 视频生成智能体 (适配 Session JSON 格式)
- 从 session.json 的 artifacts["storyboard"] 读取拍摄片段(Segments)
- 视频提示词：风格前缀 + 分镜列表(分镜1: [时长] content...)
- 参考图：从 session.json 的 artifacts["reference_generation"] 读取
- 支持逐项并发生成、实时预览、重新生成
"""

import os
import re
import glob
import json
import asyncio
import logging
from typing import Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import AgentInterface

logger = logging.getLogger(__name__)


class VideoDirectorAgent(AgentInterface):
    """视频生成：拍摄片段(Segments) → 组装提示词 → 视频片段"""

    def __init__(self):
        super().__init__(name="VideoDirector")

    # ─── 版本管理 ───

    @staticmethod
    def _video_base(sid: str) -> str:
        return os.path.join('code/result/video', str(sid))

    def _list_versions(self, sid: str, segment_id: str) -> List[str]:
        """列出某个片段视频的所有历史版本"""
        video_dir = self._video_base(sid)
        pattern = os.path.join(video_dir, f"{segment_id}*.mp4")
        files = [f for f in sorted(glob.glob(pattern), key=os.path.getmtime)
                 if not f.endswith('_final.mp4')]
        return files

    def _next_version_path(self, sid: str, segment_id: str) -> str:
        """获取下一个版本路径"""
        video_dir = self._video_base(sid)
        os.makedirs(video_dir, exist_ok=True)

        existing = self._list_versions(sid, segment_id)
        if not existing:
            return os.path.join(video_dir, f"{segment_id}.mp4")

        max_v = 1
        for fp in existing:
            bn = os.path.splitext(os.path.basename(fp))[0]
            m = re.search(r'_v(\d+)$', bn)
            if m:
                max_v = max(max_v, int(m.group(1)))

        return os.path.join(video_dir, f"{segment_id}_v{max_v + 1}.mp4")

    # ─── 视频生成 ───

    def _generate_one(self, sid: str, segment_id: str, prompt: str,
                      img_path: str, video_model: str,
                      duration: int = 10, sound: str = "",
                      shot_type: str = "multi",
                      video_ratio: str = "16:9") -> tuple:
        """生成单个视频片段，返回 (segment_id, path_or_None)"""
        if self.cancellation_check and self.cancellation_check():
            logger.info(f"VideoDirectorAgent: {segment_id} 跳过（用户取消）")
            return segment_id, None

        if not os.path.exists(img_path):
            logger.warning(f"Image missing for {segment_id}: {img_path}")
            return segment_id, None

        save_path = self._next_version_path(sid, segment_id)
        try:
            from models.video_client import VideoClient
            client = VideoClient()
            client.generate_video(
                prompt=prompt,
                image_path=img_path,
                save_path=save_path,
                model=video_model,
                duration=duration,
                sound=sound,
                shot_type=shot_type,
                video_ratio=video_ratio,
            )
            return segment_id, save_path
        except Exception as e:
            logger.error(f"Video gen failed for {segment_id}: {e}")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
        return segment_id, None

    # ─── 提示词组装 ───

    def _assemble_prompt(self, segment: dict, style_prompt: str, video_data: Optional[dict]=None) -> str:
        """组装视频提示词
        格式：
        风格控制：用户选择的风格提示词, 电影质感
        分镜列表：分镜1:[时长] content... 分镜2:[时长] content...
        """
        # 如果 video_data 中有 description，则优先使用
        # 前端修改后的提示词会存入 artifacts.video_generation.clips.description
        if video_data and "description" in video_data:
            return video_data["description"]

        # 否则从 segment 中组装
        prompt = f"风格控制：{style_prompt}\n"
        prompt += "分镜列表："
        
        shots = segment.get("shots", [])
        for i, shot in enumerate(shots, 1):
            dur = shot.get("duration", 5)
            content = shot.get("content", "").strip()
            prompt += f"\n分镜{i}：[{dur}秒] {content}"

        prompt += "\n不要生成字幕或水印"

        return prompt

    def _get_style_keywords(self, session_data: dict) -> str:
        """从会话数据获取风格关键词"""
        style = session_data.get('style', 'realistic').lower()
        
        STYLE_MAP = {
            "anime": "anime style, vibrant colors, clean lines,",
            "realistic": "photorealistic, cinematic lighting, high-detail textures,",
            "cartoon": "cartoon style, thick outlines, bold colors,",
            "3d-disney": "3D CGI animation, Disney/Pixar style, smooth textures,",
            "oil-painting": "oil painting, artistic brushstrokes, rich textures,",
            "chinese-ink": "Chinese ink wash painting, traditional style, soft strokes,"
        }
        return STYLE_MAP.get(style, "cinematic, high quality,")

    # ─── 参考图获取 ───

    def _get_reference_image(self, sid: str, segment_id: str, scene_map: dict) -> str:
        """获取参考图路径：优先用选中的版本，次之用最新版本"""
        # 1. 检查 session 中 artifacts.reference_generation.scenes 里的 selected
        if segment_id in scene_map and scene_map[segment_id].get("selected"):
            path = scene_map[segment_id]["selected"]
            if os.path.exists(path):
                return path

        # 2. 回退：扫描磁盘 Scenes 目录
        from .reference_agent import ReferenceGeneratorAgent
        versions = ReferenceGeneratorAgent._list_versions_static(sid, segment_id)
        if versions:
            return versions[-1]

        # 3. 默认路径
        return os.path.abspath(os.path.join('code/result/image', str(sid), 'Scenes', f"{segment_id}.jpg"))

    # ─── 预览 / Payload ───

    def _build_preview(self, sid: str, segments: list, scene_map: dict, style_prompt: str = "") -> list:
        preview = []
        for idx, seg in enumerate(segments, 1):
            segment_id = seg["segment_id"]
            versions = self._list_versions(sid, segment_id)
            ep_n = seg.get('episode_number', 1)
            seg_n = seg.get('segment_number', idx)
            preview.append({
                "id": segment_id,
                "name": f"第{ep_n}集-片段{seg_n}",
                "episode": ep_n,
                "index": seg_n,
                "description": self._assemble_prompt(seg, style_prompt),
                "duration": seg.get('total_duration', 10),
                "selected": versions[-1] if versions else "",
                "versions": versions,
                "status": "done" if versions else "pending",
            })
        return preview

    def _build_payload(self, sid: str, segments: list, style_prompt: str = "") -> dict:
        clips = []
        for idx, seg in enumerate(segments, 1):
            segment_id = seg["segment_id"]
            versions = self._list_versions(sid, segment_id)
            ep_n = seg.get('episode_number', 1)
            seg_n = seg.get('segment_number', idx)
            clips.append({
                "id": segment_id,
                "name": f"第{ep_n}集-片段{seg_n}",
                "episode": ep_n,
                "index": seg_n,
                "description": self._assemble_prompt(seg, style_prompt),
                "duration": seg.get('total_duration', 10),
                "selected": versions[-1] if versions else "",
                "versions": versions,
                "status": "done" if versions else "failed",
            })
        return {
            "payload": {
                "session_id": sid,
                "clips": clips,
            },
            "stage_completed": True,
        }

    def _update_session_video_data(self, sid: str, segments: list, style_prompt: str) -> None:
        """同步视频生成结果到 session.json 的 artifacts 中，适配前端读取"""
        session_path = os.path.join('code/data/sessions', f'{sid}.json')
        if not os.path.exists(session_path):
            return

        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            payload = self._build_payload(sid, segments, style_prompt)
            data.setdefault("artifacts", {})["video_generation"] = payload["payload"]
            
            # 同时保留一份给后端的 scene2video (如果需要)
            # data[sid]["scene2video"] = ... (可选)
            
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update session video data: {e}")

    # ─── 核心流程 ───

    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        from config import settings
        
        input_data = self._merge_session_params(input_data)
        sid = input_data["session_id"]
        
        # ═══ 介入：用户选择指定版本 ═══
        if intervention and "selected_clips" in intervention:
            selected_clips = intervention["selected_clips"] # Dict[segment_id, path]
            logger.info(f"[VideoAgent] 用户更新片段选择: {selected_clips}")
            
            session_path = os.path.join('code/data/sessions', f'{sid}.json')
            if os.path.exists(session_path):
                with open(session_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                clips = data.get("artifacts", {}).get("video_generation", {}).get("clips", [])
                for clip in clips:
                    cid = clip.get("id")
                    if cid in selected_clips:
                        clip["selected"] = selected_clips[cid]
                
                with open(session_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            
            # 返回当前状态
            artifacts = data.get("artifacts", {})
            episodes = artifacts.get('storyboard', {}).get('episodes', [])
            segments = []
            for ep in episodes:
                segments.extend(ep.get("segments", []))
            return self._build_payload(sid, segments)

        video_model = self._require_input(input_data, "video_model")
        enable_concurrency = input_data.get("enable_concurrency", True)
        from models.config_model import get_max_concurrency
        concurrency = get_max_concurrency(video_model, enable_concurrency)
        
        video_ratio = input_data.get("video_ratio", "16:9")
        video_shot_type = input_data.get("video_shot_type", "multi")
        
        # 加载会话数据
        session_path = os.path.join('code/data/sessions', f'{sid}.json')
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            
        artifacts = session_data.get("artifacts", {})
        
        # 1. 获取拍摄片段列表 (从 Storyboard)
        episodes = artifacts.get('storyboard', {}).get('episodes', [])
        segments = []
        for ep in episodes:
            segments.extend(ep.get("segments", []))
        if not segments:
            raise Exception("未找到分镜片段数据，请先完成阶段3")
        
        video_clips = artifacts.get('video_generation', {}).get('clips', [])

        # 2. 获取参考图路径映射 (从 Reference Generation)
        ref_art = artifacts.get('reference_generation', {})
        scene_list = ref_art.get('scenes', [])
        scene_map = {s['id']: s for s in scene_list if 'id' in s}
        
        style_zh = session_data.get('style', 'realistic')
        # 简单映射为中文显示名
        style_map_zh = {
            "anime": "动漫",
            "realistic": "写实",
            "cartoon": "卡通",
            "3d-disney": "3D迪斯尼",
            "oil-painting": "油画",
            "chinese-ink": "国画",
            "comic-book": "美漫",
            "cyberpunk": "赛博朋克"
        }
        style_name = style_map_zh.get(style_zh, style_zh)
        style_prompt = self._get_style_prompt(style_zh)

        # ═══ 介入：重新生成指定片段 ═══
        if intervention:
            regen_ids = intervention.get("regenerate_clips", [])
            if regen_ids:
                self._report_progress("视频生成", "重新生成中...", 5)
                segment_map = {s['segment_id']: s for s in segments}
                clip_map = {c['id']: c for c in video_clips}
                
                def regen_run():
                    done = 0
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futs = {}
                        for seg_id in regen_ids:
                            seg = segment_map.get(seg_id)
                            clip = clip_map.get(seg_id) if clip_map.get(seg_id) else None
                            if not seg: continue
                            prompt = self._assemble_prompt(seg, style_prompt, video_data=clip)

                            print("#"*100)
                            print(clip)
                            print(prompt)
                            return

                            img_path = self._get_reference_image(sid, seg_id, scene_map)
                            duration = seg.get("total_duration", 10)
                            fut = executor.submit(
                                self._generate_one, sid, seg_id, prompt,
                                img_path, video_model, duration,
                                "", video_shot_type, video_ratio
                            )
                            futs[fut] = seg_id
                        for fut in as_completed(futs):
                            sid_done = futs[fut]
                            try:
                                _, res_path = fut.result()
                            except Exception as e:
                                logger.error(f"Regen future error for {sid_done}: {e}")
                                res_path = None
                            done += 1
                            pct = 5 + int(90 * done / max(1, len(regen_ids)))
                            if res_path:
                                versions = self._list_versions(sid, sid_done)
                                self._report_progress("视频生成", f"完成: {sid_done}", pct, data={
                                    "asset_complete": {
                                        "type": "clips", "id": sid_done,
                                        "status": "done",
                                        "selected": res_path,
                                        "versions": versions,
                                    }
                                })
                            else:
                                self._report_progress("视频生成", f"失败: {sid_done}", pct, data={
                                    "asset_complete": {
                                        "type": "clips", "id": sid_done,
                                        "status": "failed",
                                        "selected": "", "versions": [],
                                    }
                                })
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, regen_run)
                
                # 同步到 session artifacts
                self._update_session_video_data(sid, segments, style_prompt)
                
                return self._build_payload(sid, segments, style_prompt)

        # ═══ 正常流程：全量生成 ═══
        self._report_progress("视频生成", "正在准备数据...", 2)
        preview = self._build_preview(sid, segments, scene_map, style_prompt)
        self._report_progress("视频生成", "加载视频列表", 5, data={"assets_preview": {"clips": preview}})

        def run():
            tasks = []
            for seg in segments:
                seg_id = seg["segment_id"]
                existing = self._list_versions(sid, seg_id)
                if existing: continue
                prompt = self._assemble_prompt(seg, style_prompt)
                img_path = self._get_reference_image(sid, seg_id, scene_map)
                duration = seg.get("total_duration", 10)
                tasks.append((seg_id, prompt, img_path, duration))
            if not tasks:
                self._report_progress("视频生成", "所有视频片段已存在", 95)
                return
            done = 0
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futs = {}
                for seg_id, prompt, img_path, dur in tasks:
                    # 提交前立即发送正在运行的状态，让前端 UI 更新
                    self._report_progress("视频生成", f"启动生成: {seg_id}", 5, data={
                        "asset_complete": {
                            "type": "clips", "id": seg_id,
                            "status": "running"
                        }
                    })
                    fut = executor.submit(
                        self._generate_one, sid, seg_id, prompt,
                        img_path, video_model, dur,
                        "", video_shot_type, video_ratio
                    )
                    futs[fut] = seg_id
                for fut in as_completed(futs):
                    sid_done = futs[fut]
                    try:
                        _, res_path = fut.result()
                    except Exception as e:
                        logger.error(f"Video future error for {sid_done}: {e}")
                        res_path = None
                    done += 1
                    pct = 5 + int(90 * done / max(1, len(tasks)))
                    if res_path:
                        versions = self._list_versions(sid, sid_done)
                        self._report_progress("视频生成", f"完成: {sid_done}", pct, data={
                            "asset_complete": {
                                "type": "clips", "id": sid_done,
                                "status": "done",
                                "selected": res_path,
                                "versions": versions,
                            }
                        })
                    else:
                        self._report_progress("视频生成", f"失败: {sid_done}", pct, data={
                            "asset_complete": {
                                "type": "clips", "id": sid_done,
                                "status": "failed",
                                "selected": "", "versions": [],
                            }
                        })
                    if self.cancellation_check and self.cancellation_check():
                        for f in futs:
                            if not f.done(): f.cancel()
                        break
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run)
        
        # 同步到 session artifacts
        self._update_session_video_data(sid, segments, style_prompt)
        
        self._report_progress("视频生成", "完成", 100)
        return self._build_payload(sid, segments, style_prompt)
