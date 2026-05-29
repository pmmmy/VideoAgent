# -*- coding: utf-8 -*-
"""
阶段4: 参考图生成智能体
- 基于阶段3分镜(shots)，为每个分镜生成「首帧图像提示词」，再据此生成参考图
- 首帧提示词由 LLM 根据 shot 的 plot、visual_prompt、duration 生成
- 阶段5生视频时使用阶段3的原始分镜描述，而非首帧提示词
- 支持逐项实时预览、重新生成、多版本管理
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
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)


def ratio_to_size(ratio: str) -> str:
    """将视频比例转换为图像尺寸"""
    size_map = {
        "16:9": "1920*1080",
        "9:16": "1080*1920",
        "1:1": "1024*1024",
        "4:3": "1024*768",
        "3:4": "768*1024",
        "21:9": "2560*1080",
    }
    return size_map.get(ratio, "1920*1080")


class ReferenceGeneratorAgent(AgentInterface):
    """参考图生成：分镜(阶段3) → 首帧提示词(LLM) → 参考图(图像模型)"""

    def __init__(self):
        super().__init__(name="ReferenceGenerator")

    # ─── 版本管理 ───

    @staticmethod
    def _scenes_base(sid: str) -> str:
        return os.path.join('code/result/image', str(sid), 'Scenes')

    def _list_versions(self, sid: str, shot_id: str) -> List[str]:
        """列出某个分镜的所有历史版本
        命名: shot_001_01.jpg, shot_001_01_v2.jpg, ...
        """
        return self._list_versions_static(sid, shot_id)

    @staticmethod
    def _list_versions_static(sid: str, shot_id: str) -> List[str]:
        """列出某个分镜的所有历史版本（静态方法，供外部调用）"""
        scenes_dir = os.path.join('code/result/image', str(sid), 'Scenes')
        pattern = os.path.join(scenes_dir, f"{shot_id}*.jpg")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        return files

    def _next_version_path(self, sid: str, shot_id: str) -> str:
        """获取下一个版本路径"""
        scenes_dir = self._scenes_base(sid)
        os.makedirs(scenes_dir, exist_ok=True)

        existing = self._list_versions(sid, shot_id)
        if not existing:
            return os.path.join(scenes_dir, f"{shot_id}.jpg")

        max_v = 1
        for fp in existing:
            bn = os.path.splitext(os.path.basename(fp))[0]
            m = re.search(r'_v(\d+)$', bn)
            if m:
                max_v = max(max_v, int(m.group(1)))

        return os.path.join(scenes_dir, f"{shot_id}_v{max_v + 1}.jpg")

    # ─── 素材匹配 ───

    @staticmethod
    def _build_asset_map(character_design: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """从阶段2生成的素材数据中构建映射，不再直接扫描磁盘"""
        am: Dict[str, Dict[str, str]] = {'characters': {}, 'settings': {}}
        
        # 处理角色
        for char in character_design.get('characters', []):
            cid = char.get('id') or char.get('character_id')
            selected = char.get('selected')
            if cid and selected:
                am['characters'][cid] = selected
                
        # 处理场景
        for setting in character_design.get('settings', []):
            sid = setting.get('id') or setting.get('setting_id')
            selected = setting.get('selected')
            if sid and selected:
                am['settings'][sid] = selected
                
        return am

    def _collect_refs(self, segment: dict, asset_map: dict,
                      char_id_map: dict, setting_id_map: dict) -> List[str]:
        """为一个片段(Segment)收集参考原图路径（角色 + 场景素材）"""
        refs = []
        # 1. 角色匹配
        for cn in segment.get('characters', []):
            cid = char_id_map.get(cn)
            # 如果名称不直接匹配，尝试模糊匹配（部分包含）
            if not cid:
                for name, _id in char_id_map.items():
                    if name in cn or cn in name:
                        cid = _id
                        break
            
            if cid and cid in asset_map['characters']:
                refs.append(os.path.abspath(asset_map['characters'][cid]))
                logger.info(f"[{segment.get('segment_id', '')}] 添加角色参考图: {cn} -> {cid}")

        # 2. 场景匹配
        loc = segment.get('location', '')
        set_id = setting_id_map.get(loc)
        # 如果名称不直接匹配，尝试模糊匹配
        if not set_id and loc:
            for name, _id in setting_id_map.items():
                if name in loc or loc in name:
                    set_id = _id
                    logger.info(f"[{segment.get('segment_id', '')}] 场景模糊匹配成功: {loc} -> {name} ({set_id})")
                    break

        if set_id and set_id in asset_map['settings']:
            refs.append(os.path.abspath(asset_map['settings'][set_id]))
            logger.info(f"[{segment.get('segment_id', '')}] 添加场景参考图: {loc} -> {set_id}")
        else:
            logger.warning(f"[{segment.get('segment_id', '')}] 未找到场景参考图: location={loc}, set_id={set_id}, available_settings={list(asset_map['settings'].keys())}")
        
        logger.info(f"[{segment.get('segment_id', '')}] 共收集 {len(refs)} 张参考图")
        return refs[:10]

    def _get_descriptions(self, segment: dict, char_id_map: dict, setting_id_map: dict,
                          character_json: dict) -> tuple:
        """获取片段中涉及的角色和场景描述

        Returns:
            (character_description, setting_description)
        """
        # 角色描述
        char_descs = []
        for cn in segment.get('characters', []):
            cid = char_id_map.get(cn, '')
            if cid:
                for c in character_json.get('characters', []):
                    if (c.get('id') or c.get('character_id')) == cid:
                        desc = c.get('description', '')
                        if desc:
                            char_descs.append(f"{cn}: {desc}")
                        break

        # 场景描述
        loc = segment.get('location', '')
        set_id = setting_id_map.get(loc)
        setting_desc = ""
        if set_id:
            for s in character_json.get('settings', []):
                if (s.get('id') or s.get('setting_id')) == set_id:
                    setting_desc = s.get('description', '')
                    break

        return "； ".join(char_descs), setting_desc

    # ─── 首帧提示词生成 ───

    # ─── 预览构建 ───

    def _build_preview(self, sid: str, segments: list, session_data: dict = None) -> list:
        """构建片段预览列表（含当前状态）"""
        preview = []

        # 建立 scene_id 到 selected 路径的映射
        selected_map = {}
        if session_data and "artifacts" in session_data:
            ref_gen = session_data["artifacts"].get("reference_generation", {})
            for scene in ref_gen.get("scenes", []):
                sid_in_json = scene.get("id")
                if sid_in_json:
                    selected_map[sid_in_json] = scene.get("selected", "")

        for idx, seg in enumerate(segments, 1):
            segment_id = seg.get('segment_id', f'seg_unk_{idx}')
            versions = self._list_versions(sid, segment_id)

            # 优先从 artifacts 中读取 selected 字段，如果没有则回退到最后一个版本
            selected_path = selected_map.get(segment_id)
            if not selected_path:
                selected_path = versions[-1] if versions else ""

            # 获取该段下第一个镜头的 content 作为片段描述
            plot = seg.get('shots', [])[0].get('content', '') if seg.get('shots') else ""
            ep_n = seg.get('episode_number', 1)
            seg_n = seg.get('segment_number', idx)

            preview.append({
                "id": segment_id,
                "name": f"第{ep_n}集-片段{seg_n}",
                "episode": ep_n,
                "index": seg_n,
                "description": plot,
                "selected": selected_path,
                "versions": versions,
                "status": "done" if versions else "pending",
            })
        return preview

    # ─── 单张生成 ───

    def _generate_one(self, img_client, sid: str, segment: dict,
                      first_frame_prompt: str, refs: List[str],
                      style: str, it2i_model: str, t2i_model: str,
                      video_ratio: str = "16:9", resolution: str = "1080P", vlm_model: str = "qwen3.5-plus",
                      character_description: str = "", setting_description: str = "",
                      max_versions: int = 3) -> tuple:
        """生成单个片段参考图，返回 (segment_id, path_or_None, eval_result)

        最多生成 max_versions 个版本，如果所有版本都没有达到8分，
        使用 VLM 选择最好的一张作为最终参考图。
        """
        segment_id = segment.get('segment_id', '')

        # 仅提取第一个镜头的描述作为当前 Plot
        plot = segment.get('shots', [])[0].get('content', '') if segment.get('shots') else ""
        visual_prompt = first_frame_prompt

        # 取消时直接跳过，不抛异常，以保留已生成的部分结果
        if self.cancellation_check and self.cancellation_check():
            logger.info(f"ReferenceGeneratorAgent: {segment_id} 跳过（用户取消）")
            return segment_id, None, None

        model = it2i_model if refs else t2i_model
        logger.info(f"[{segment_id}] 使用模型: {model}, 参考图数量: {len(refs) if refs else 0}")
        if refs:
            for i, r in enumerate(refs):
                logger.info(f"[{segment_id}] 参考图[{i}]: {r}")

        # 收集所有生成的版本
        all_versions = []
        all_eval_results = []

        for version in range(max_versions):
            self._check_cancel()

            style_prompt = self._get_style_prompt(style)
            full_prompt = f"{style_prompt}, {first_frame_prompt}"

            save_path = self._next_version_path(sid, segment_id)
            save_dir = os.path.dirname(save_path)

            try:
                paths = img_client.generate_image(
                    prompt=full_prompt,
                    image_paths=refs if refs else None,
                    model=model,
                    session_id=str(sid),
                    save_dir=save_dir,
                    video_ratio=video_ratio,
                    resolution=resolution,
                )
                if not paths:
                    continue

                gen = paths[0]
                if gen != save_path:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    os.rename(gen, save_path)

                # VLM 评估
                eval_result = self._evaluate_with_vlm(save_path, segment, plot, visual_prompt,
                                                      character_description=character_description,
                                                      setting_description=setting_description,
                                                      vlm_model=vlm_model)

                score = eval_result.get('score', 0)
                is_acceptable = score >= 8

                logger.info(f"[{segment_id}] 版本{version + 1}: 评分 {score}/10, {'✓通过' if is_acceptable else '✗不通过'}")

                # 记录版本信息
                all_versions.append(save_path)
                all_eval_results.append(eval_result)

                # 如果达到8分，立即返回
                if is_acceptable:
                    return segment_id, save_path, eval_result

                # 报告进度
                if version < max_versions - 1:
                    self._report_progress("参考图", f"重新生成中 ({version + 2}/{max_versions}): {segment_id}", 0)

            except Exception as e:
                logger.error(f"Segment {segment_id} image generation failed: {e}")

        # 所有版本都没有达到8分，使用 VLM 选择最好的
        if all_versions:
            logger.warning(f"[{segment_id}] 所有版本都未达到8分，使用VLM选择最佳...")
            best_path, best_eval = self._select_best_with_vlm(
                all_versions, segment, plot, visual_prompt,
                character_description=character_description,
                setting_description=setting_description,
                vlm_model=vlm_model
            )
            if best_path:
                return segment_id, best_path, best_eval

        # 如果没有任何生成成功
        logger.warning(f"[{segment_id}] 没有成功生成任何图片")
        return segment_id, None, None

    def _select_best_with_vlm(self, image_paths: List[str], segment: dict, plot: str, visual_prompt: str,
                              character_description: str = "", setting_description: str = "",
                              vlm_model: str = "qwen3.5-plus") -> tuple:
        """使用 VLM 从多个版本中选择最好的一张"""
        from models.vlm_client import VLM

        if not image_paths:
            return None, None

        segment_id = segment.get('segment_id', '')

        # 加载评估提示词
        select_prompt = load_prompt('reference', 'eval_select_best', 'zh').format(
            num_images=len(image_paths),
            num_images_minus_1=len(image_paths) - 1,
            plot=plot,
            visual_prompt=visual_prompt,
            character_description=character_description,
            setting_description=setting_description,
            images_list="\n".join([f"图片{i}: {p}" for i, p in enumerate(image_paths)])
        )

        try:
            vlm = VLM()
            result = vlm.query(select_prompt, image_paths=image_paths, model=vlm_model)
            logger.info(f"[{segment_id}] VLM选择结果: {result}")

            # 解析 JSON 结果
            import re
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                selected = json.loads(json_match.group())
                selected_idx = selected.get('selected_index', 0)
                if 0 <= selected_idx < len(image_paths):
                    best_path = image_paths[selected_idx]
                    logger.info(f"[{segment_id}] VLM选择第{selected_idx + 1}张作为最佳图片")
                    # 构建评估结果
                    best_eval = {
                        "score": selected.get('score', 5),
                        "issues": selected.get('issues', []),
                        "is_acceptable": True,
                        "selected_by_vlm": True,
                        "reason": selected.get('reason', '')
                    }
                    return best_path, best_eval

        except Exception as e:
            logger.error(f"[{segment_id}] VLM选择最佳图片失败: {e}")

        # 如果失败，返回第一个版本
        return image_paths[0], {"score": 5, "issues": [], "selected_by_vlm": False}

    def _evaluate_with_vlm(self, image_path: str, segment: dict, plot: str, visual_prompt: str,
                          character_description: str = "", setting_description: str = "",
                          vlm_model: str = "qwen3.5-plus") -> dict:
        """使用 VLM 评估首帧参考图"""
        try:
            from models.vlm_client import VLM
            vlm = VLM()

            eval_prompt = load_prompt('reference', 'eval_first_frame', 'zh').format(
                plot=plot,
                visual_prompt=visual_prompt,
                character_description=character_description,
                setting_description=setting_description
            )

            result = vlm.query(
                prompt=eval_prompt,
                image_paths=[image_path],
                model=vlm_model
            )

            if result and isinstance(result, list):
                result_text = result[0] if result else ""
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = str(result)

            import json
            try:
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    eval_result = json.loads(json_match.group())
                    return eval_result
            except:
                pass

            return {"score": 5, "issues": ["评估解析失败"], "is_acceptable": True}

        except Exception as e:
            logger.warning(f"VLM evaluation failed: {e}")
            return {"score": 5, "issues": [str(e)], "is_acceptable": True}

    # ─── 构建最终 payload ───

    def _build_payload(self, sid: str, segments: list, session_data: dict = None, prompts_map: dict = None, selected_images: dict = None) -> dict:
        """构建最终 payload"""
        scenes = []
        if prompts_map is None:
            prompts_map = {}
        if selected_images is None:
            selected_images = {}

        # 建立 scene_id 到 selected 路径的映射
        selected_map = {}
        existing_prompts = {}
        if session_data and "artifacts" in session_data:
            ref_gen = session_data["artifacts"].get("reference_generation", {})
            for scene in ref_gen.get("scenes", []):
                sid_in_json = scene.get("id")
                if sid_in_json:
                    selected_map[sid_in_json] = scene.get("selected", "")
                    existing_prompts[sid_in_json] = scene.get("visual_prompt", "")

        for idx, seg in enumerate(segments, 1):
            segment_id = seg.get('segment_id', f'seg_unk_{idx}')
            versions = self._list_versions(sid, segment_id)
            
            # 优先从本轮生成的 selected_images 中读取，如果找不到，再从 session 的 artifacts 中读取 selected 字段
            selected_path = selected_images.get(segment_id)
            if not selected_path:
                selected_path = selected_map.get(segment_id)
            if not selected_path:
                selected_path = versions[-1] if versions else ""

            # 获取该段下第一个镜头的 content 作为片段描述
            shots_summary = seg.get('shots', [])[0].get('content', '') if seg.get('shots') else ""
            
            # 取提示词
            visual_prompt = prompts_map.get(segment_id) or existing_prompts.get(segment_id) or ""

            # status: 有图片=done, 无图片=pending(待生成)
            status = "done" if versions else "pending"
            scenes.append({
                "id": segment_id,
                "name": f"第{seg.get('episode_number', 1)}集-片段{seg.get('segment_number', idx)}",
                "index": idx,
                "description": shots_summary,
                "visual_prompt": visual_prompt,
                "selected": selected_path,
                "versions": versions,
                "status": status,
            })
        return {
            "payload": {
                "session_id": sid,
                "scenes": scenes,
            },
            "stage_completed": True,
        }

    # ─── 核心流程 ───

    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        from config import settings
        from models.image_client import ImageClient
        from models.llm_client import LLM

        # 从 session.json 补齐缺失的参数（关键修复：防止前端漏传导致的默认值回退）
        input_data = self._merge_session_params(input_data)

        sid = input_data["session_id"]
        
        style = input_data.get("style", "anime")
        video_ratio = input_data.get("video_ratio", "16:9")
        resolution = input_data.get("resolution", "2K")
        llm_model = self._require_input(input_data, "llm_model")
        t2i = self._require_input(input_data, "image_t2i_model")
        it2i = self._require_input(input_data, "image_it2i_model")
        vlm_model = self._require_input(input_data, "vlm_model")
        # 根据 enable_concurrency 决定并发数
        enable_concurrency = input_data.get("enable_concurrency", True)
        logger.info(f"[ReferenceAgent] enable_concurrency={enable_concurrency}")
        # 取 t2i 和 it2i 中的最大并发数
        from models.config_model import get_max_concurrency
        max_t2i = get_max_concurrency(t2i, enable_concurrency)
        max_it2i = get_max_concurrency(it2i, enable_concurrency)
        concurrency = max(max_t2i, max_it2i)
        logger.info(f"[ReferenceAgent] 使用并发数={concurrency}")

        # 读取会话数据
        session_path = os.path.join('code/data/sessions', f'{sid}.json')
        if not os.path.exists(session_path):
            raise Exception(f"Session file not found: {session_path}")
            
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            
        artifacts = session_data.get("artifacts", {})
        
        # 提取已经存在于 session 中的 visual_prompts
        session_visual_prompts = {}
        ref_gen = artifacts.get("reference_generation", {})
        for scene in ref_gen.get("scenes", []):
            sid_in_json = scene.get("id")
            vp = scene.get("visual_prompt")
            if sid_in_json and vp:
                session_visual_prompts[sid_in_json] = vp

        img_client = ImageClient(
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            dashscope_base_url=settings.DASHSCOPE_BASE_URL,
            gpt_api_key=settings.OPENAI_API_KEY,
            gpt_base_url=settings.OPENAI_BASE_URL,
            proxy=settings.provider_proxy("openai"),
            ark_api_key=settings.ARK_API_KEY,
            ark_base_url=settings.ARK_BASE_URL,
        )

        episodes = artifacts.get('storyboard', {}).get('episodes', [])
        if not episodes:
            raise Exception("未找到分镜剧集数据，请先完成阶段3")

        segments = []
        for ep in episodes:
            for seg in ep.get("segments", []):
                segments.append(seg)
                
        if not segments:
            raise Exception("未找到分镜片段数据，请先完成阶段3")

        logger.info(f"[ReferenceAgent] 解析到 {len(segments)} 个拍摄片段")

        script_json = artifacts.get('script_generation', {})
        character_json = artifacts.get('character_design', {})
        result_file = os.path.join('code/data/sessions', f'{sid}.json')

        # 判断中英文
        is_zh = any('\u4e00' <= c <= '\u9fff' for c in script_json.get("title", ""))

        # 构建 name → id 映射（用于素材匹配）
        char_id_map = {}
        for c in character_json.get('characters', []):
            chara_id = c.get('id') or c.get('character_id') or ''
            char_id_map[c['name']] = chara_id

        setting_id_map = {}
        for s in character_json.get('settings', []):
            set_id = s.get('id') or s.get('setting_id') or ''
            setting_id_map[s['name']] = set_id

        asset_map = self._build_asset_map(character_json)

        # ═══ 介入：重新生成指定分段 ═══
        if intervention:
            regen_scenes = intervention.get("regenerate_scenes", [])

            if regen_scenes:
                self._report_progress("参考图", "重新生成中...", 2)

                # 重新从 session JSON 文件读取最新的 storyboard 数据
                with open(session_path, 'r', encoding='utf-8') as f:
                    fresh_data = json.load(f)
                fresh_artifacts = fresh_data.get('artifacts', {})
                fresh_episodes = fresh_artifacts.get('storyboard', {}).get('episodes', [])
                
                fresh_segments = []
                for ep in fresh_episodes:
                    fresh_segments.extend(ep.get("segments", []))
                    
                fresh_segment_map = {s['segment_id']: s for s in fresh_segments}

                llm = LLM()

                selected_images = {}
                prompt_map = {}  # segment_id → first_frame_prompt

                def regen_run():
                    total = len(regen_scenes)
                    done = 0
                    nonlocal selected_images
                    nonlocal prompt_map
                    # 每片段5个步骤：准备(1)、生成(3)、完成(1)
                    steps_per_segment = 5
                    total_steps = total * steps_per_segment

                    def calc_pct_regen(step: int) -> int:
                        return min(2 + int(98 * step / total_steps), 100)

                    # 根据最新的 Segment 生成对应的 visual_prompt
                    for i, segment_id in enumerate(regen_scenes):
                        seg = fresh_segment_map.get(segment_id, {})
                        
                        # 提取第一个分镜的剧情供首帧提示词使用（视频首帧应基于片段起点）
                        first_shot = seg.get('shots', [])[0] if seg.get('shots') else {}
                        plot = first_shot.get('content', '')
                        char_desc, set_desc = self._get_descriptions(seg, char_id_map, setting_id_map, character_json)

                        existing_vp = session_visual_prompts.get(segment_id)
                        if existing_vp:
                            ff_prompt = existing_vp
                            logger.info(f"[{segment_id}] 重新生成时命中已有提示词，复用原提示词...")
                        else:
                            ff_prompt_tpl = load_prompt('reference', 'first_frame', 'zh' if is_zh else 'en')
                            ff_prompt_resp = self._cancellable_query(
                                llm,
                                prompt=ff_prompt_tpl.format(
                                    original_text=script_json.get("original_text", ""),
                                    plot=plot,
                                    character_description=char_desc,
                                    setting_description=set_desc
                                ),
                                model=llm_model
                            )
                            if hasattr(ff_prompt_resp, 'content'):
                                ff_prompt = ff_prompt_resp.content.strip()
                            else:
                                ff_prompt = str(ff_prompt_resp).strip()
                                
                        prompt_map[segment_id] = ff_prompt
                        
                        logger.info(f"[{segment_id}] first-frame prompt: {ff_prompt}...")
                        self._report_progress("参考图", f"准备提示词: {segment_id}", calc_pct_regen(i * steps_per_segment + 1))

                    # 并发生成图像
                    self._report_progress("参考图", "生成参考图...", calc_pct_regen(total * 2))
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futs = {}
                        for segment_id in regen_scenes:
                            seg = fresh_segment_map.get(segment_id, {})
                            refs = self._collect_refs(seg, asset_map, char_id_map, setting_id_map)
                            char_desc, set_desc = self._get_descriptions(
                                seg, char_id_map, setting_id_map, character_json
                            )
                            fut = executor.submit(
                                self._generate_one, img_client, sid,
                                seg, prompt_map[segment_id], refs,
                                style, it2i, t2i, video_ratio, resolution, vlm_model,
                                character_description=char_desc, setting_description=set_desc
                            )
                            futs[fut] = segment_id
                        for fut in as_completed(futs):
                            segment_id_done = futs[fut]
                            try:
                                _, result_path, eval_result = fut.result()
                            except Exception as e:
                                logger.error(f"Regen future error for {segment_id_done}: {e}")
                                result_path = None
                            done += 1
                            step = done * steps_per_segment
                            pct = calc_pct_regen(step)
                            if result_path:
                                selected_images[segment_id_done] = result_path
                                versions = self._list_versions(sid, segment_id_done)
                                self._report_progress("参考图", f"完成: {segment_id_done}", pct, data={
                                    "asset_complete": {
                                        "type": "scenes", "id": segment_id_done,
                                        "status": "done",
                                        "selected": result_path,
                                        "versions": versions,
                                    }
                                })
                            else:
                                self._report_progress("参考图", f"失败: {segment_id_done}", pct, data={
                                    "asset_complete": {
                                        "type": "scenes", "id": segment_id_done,
                                        "status": "failed",
                                        "selected": "", "versions": [],
                                    }
                                })
                            # 检查取消
                            if self.cancellation_check and self.cancellation_check():
                                logger.info("ReferenceGeneratorAgent: 用户取消重新生成，停止等待剩余任务")
                                for f in futs:
                                    if not f.done():
                                        f.cancel()
                                break

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, regen_run)

                self._report_progress("参考图", "完成", 100)
                return self._build_payload(sid, fresh_segments, session_data, prompt_map, selected_images)

        # ═══ 正常流程：全量生成 ═══
        self._report_progress("参考图", "加载分镜数据...", 5)

        # 发送预览列表
        preview = self._build_preview(sid, segments, session_data)
        self._report_progress("参考图", "加载分镜列表", 8, data={"assets_preview": {"scenes": preview}})

        llm = LLM()
        first_frame_prompts = {}  # 提升作用域，用于最后写回结果文件
        selected_images_map = {}  # 提升作用域，记录本轮新生成且 VLM 挑选出来的图片路径

        def run():
            nonlocal first_frame_prompts
            nonlocal selected_images_map
            # 筛选需要生成的（跳过已有图的）
            pending_segments = []
            for seg in segments:
                segment_id = seg['segment_id']
                existing = self._list_versions(sid, segment_id)
                if existing:
                    continue
                pending_segments.append(seg)

            if not pending_segments:
                self._report_progress("参考图", "所有分镜图已存在", 95)
                return

            total = len(pending_segments)
            # 每分镜5个步骤：准备(1)、生成(3)、完成(1)
            steps_per_segment = 5
            total_steps = total * steps_per_segment + 1  # +1 是加载数据步骤

            def calc_pct(step: int) -> int:
                """根据步骤计算进度百分比"""
                return min(2 + int(98 * step / total_steps), 100)

            done = 0

            # 步骤2-6(每片段)：流式生成提示词并立即开始图像生成
            self._report_progress("参考图", "开始生成...", calc_pct(0))
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futs = {}
                done = 0
                
                for i, seg in enumerate(pending_segments):
                    segment_id = seg['segment_id']
                    
                    # 1. 准备该片段的视觉提示词（基于片段的第一个分镜）
                    first_shot = seg.get('shots', [])[0] if seg.get('shots') else {}
                    plot = first_shot.get('content', '')
                    char_desc, set_desc = self._get_descriptions(seg, char_id_map, setting_id_map, character_json)
                    
                    existing_vp = session_visual_prompts.get(segment_id)
                    if existing_vp:
                        ff_prompt = existing_vp
                        logger.info(f"[{segment_id}] 命中已有提示词，复用原提示词...")
                    else:
                        ff_prompt_tpl = load_prompt('reference', "first_frame", 'zh' if is_zh else 'en')
                        try:
                            ff_prompt_resp = self._cancellable_query(
                                llm,
                                prompt=ff_prompt_tpl.format(
                                    original_text=script_json.get("original_text", ""),
                                    plot=plot,
                                    character_description=char_desc,
                                    setting_description=set_desc
                                ),
                                model=llm_model
                            )
                            if hasattr(ff_prompt_resp, 'content'):
                                ff_prompt = ff_prompt_resp.content.strip()
                            else:
                                ff_prompt = str(ff_prompt_resp).strip()
                        except Exception as e:
                            logger.error(f"Error generating first-frame prompt for {segment_id}: {e}")
                            ff_prompt = plot[:200]
                    
                    first_frame_prompts[segment_id] = ff_prompt
                    logger.info(f"[{segment_id}] Prompt ready, starting image generation...")

                    # 2. 发送任务开始状态 (Running)，以便前端 UI 立即显示加载状态
                    self._report_progress("参考图", f"正在启动: {segment_id}", calc_pct(i * steps_per_segment), data={
                        "asset_complete": {
                            "type": "scenes", "id": segment_id,
                            "status": "running"
                        }
                    })

                    # 3. 立即提交图像生成任务，不再等待其他片段的提示词
                    refs = self._collect_refs(seg, asset_map, char_id_map, setting_id_map)
                    char_desc, set_desc = self._get_descriptions(seg, char_id_map, setting_id_map, character_json)
                    fut = executor.submit(
                        self._generate_one, img_client, sid,
                        seg, ff_prompt, refs,
                        style, it2i, t2i, video_ratio, resolution, vlm_model,
                        character_description=char_desc, setting_description=set_desc
                    )
                    futs[fut] = segment_id
                    
                    # 报告进度（提交任务也算一点进度）
                    self._report_progress("参考图", f"等待生成: {segment_id}", calc_pct(i * steps_per_segment))

                # 4. 等待所有任务完成
                cancelled = False
                for fut in as_completed(futs):
                    segment_id_done = futs[fut]
                    try:
                        _, result_path, eval_result = fut.result()
                    except Exception as e:
                        logger.error(f"Image future error for {segment_id_done}: {e}")
                        result_path = None
                    
                    done += 1
                    step = done * steps_per_segment
                    pct = calc_pct(step)
                    
                    if result_path:
                        selected_images_map[segment_id_done] = result_path
                        versions = self._list_versions(sid, segment_id_done)
                        self._report_progress("参考图", f"完成: {segment_id_done}", pct, data={
                            "asset_complete": {
                                "type": "scenes", "id": segment_id_done,
                                "status": "done",
                                "selected": result_path,
                                "versions": versions,
                            }
                        })
                    else:
                        self._report_progress("参考图", f"失败: {segment_id_done}", pct, data={
                            "asset_complete": {
                                "type": "scenes", "id": segment_id_done,
                                "status": "failed",
                                "selected": "", "versions": [],
                            }
                        })
                    
                    # 检查取消
                    if self.cancellation_check and self.cancellation_check():
                        logger.info("ReferenceGeneratorAgent: 用户取消，停止等待剩余任务")
                        for f in futs:
                            if not f.done():
                                f.cancel()
                        cancelled = True
                        break

            if cancelled:
                self._report_progress("参考图", "已取消（保留已完成图片）", 96)
            else:
                self._report_progress("参考图", "保存结果...", 96)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, run)
        except Exception as e:
            if "cancel" in str(e).lower():
                logger.info("ReferenceGeneratorAgent: 用户取消，返回已完成部分结果")
                self._report_progress("参考图", "已取消（保留已完成图片）", 100)
                return self._build_payload(sid, segments, session_data, first_frame_prompts, selected_images_map)
            raise

        self._report_progress("参考图", "完成", 100)
        return self._build_payload(sid, segments, session_data, first_frame_prompts, selected_images_map)
