# -*- coding: utf-8 -*-
"""
阶段3: 分镜智能体
基于剧本JSON，逐场景拆分为带时长标签的分镜（shots），按幕分组输出。
支持 Segment -> Shots 嵌套结构。
"""

import os
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, Dict, List

from .base_agent import AgentInterface

logger = logging.getLogger(__name__)

def _get_shot_prompt(lang: str = "zh") -> str:
    from prompts.loader import load_prompt_with_fallback
    return load_prompt_with_fallback("storyboard", "shot", lang, "zh")

class StoryboardAgent(AgentInterface):
    def __init__(self):
        super().__init__(name="Storyboard")

    @staticmethod
    def _read_script_json(sid: str) -> dict:
        session_path = os.path.join("code/data/sessions", f"{sid}.json")
        if not os.path.exists(session_path):
            return {}
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("artifacts", {}).get("script_generation", {})
        except Exception:
            return {}

    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[dict]]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
            if isinstance(result, list): return result
        except json.JSONDecodeError: pass
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list): return result
            except json.JSONDecodeError: pass
        return None

    @staticmethod
    def _validate_episodes(episodes: List[dict]) -> List[dict]:
        """验证嵌套的 Episode -> Segment -> Shots 结构"""
        valid_episodes = []
        for ep in episodes:
            if not isinstance(ep, dict): continue
            
            segments = ep.get("segments", [])
            valid_segments = []
            for idx, seg in enumerate(segments, 1):
                if not isinstance(seg, dict): continue
                
                shots = seg.get("shots", [])
                valid_shots = []
                calc_total_duration = 0
                
                for s in shots:
                    if not isinstance(s, dict): continue
                    dur = s.get("duration", 5)
                    calc_total_duration += dur
                    valid_shots.append({
                        "shot_number": s.get("shot_number", len(valid_shots) + 1),
                        "shot_type": s.get("shot_type", "中景"),
                        "duration": dur,
                        "content": s.get("content", "")
                    })
                
                valid_segments.append({
                    "segment_id": seg.get("segment_id", f"seg_{str(idx).zfill(8)}"),
                    "segment_number": seg.get("segment_number", len(valid_segments) + 1),
                    "total_duration": seg.get("total_duration", calc_total_duration),
                    "location": seg.get("location", ""),
                    "characters": seg.get("characters", []),
                    "shots": valid_shots
                })
            
            valid_episodes.append({
                "episode_number": ep.get("episode_number", len(valid_episodes) + 1),
                "episode_title": ep.get("episode_title", ""),
                "segments": valid_segments
            })
        return valid_episodes

    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        from models.llm_client import LLM
        input_data = self._merge_session_params(input_data)
        sid = input_data.get("session_id")
        if not sid: raise Exception("Missing session_id")
             
        session_file = os.path.join("code/data/sessions", f"{sid}.json")
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
            
        llm_model = input_data.get("llm_model") or session_data.get("llm_model")
        if not llm_model:
            raise ValueError("Missing required model configuration: llm_model")
        style = input_data.get("style") or session_data.get("style") or "anime"
        
        # 处理人工干预/修改
        if intervention and "modified_storyboard" in intervention:
            modified_episodes = intervention["modified_storyboard"]
            if isinstance(modified_episodes, str): modified_episodes = json.loads(modified_episodes)
            session_data.setdefault("artifacts", {})["storyboard"] = {
                "session_id": sid,
                "episodes": modified_episodes,
                "user_modified": True,
                "updated_at": datetime.now().isoformat()
            }
            # 更新顶层状态
            session_data["updated_at"] = datetime.now().timestamp()
            with open(session_file, "w", encoding="utf-8") as f: json.dump(session_data, f, indent=2, ensure_ascii=False)
            return {"payload": {"session_id": sid, "episodes": modified_episodes}, "stage_completed": True}
        
        script_data = self._read_script_json(sid)
        if not script_data: raise Exception("未找到剧本数据")
        
        episodes = script_data.get("episodes", [])
        if not episodes:
            raise Exception("剧本数据中不包含有效集数列表(episodes)")

        # 检查是否有已存在的分镜数据，识别需要生成的集数
        existing_storyboard = session_data.get("artifacts", {}).get("storyboard", {})
        existing_story_eps = existing_storyboard.get("episodes", [])
        
        # 建立已生成的 segments 索引
        ready_eps = {e["episode_number"] for e in existing_story_eps if e.get("segments")}
        
        # 确定需要处理的集数：如果该集还没有 segments，则需要生成
        episodes_to_proc = [ep for ep in episodes if ep.get("episode_number") not in ready_eps]
        
        if not episodes_to_proc:
            logger.info("[Storyboard] All episodes already have storyboard segments. Skipping generation.")
            return {"payload": {"session_id": sid, "episodes": existing_story_eps}, "stage_completed": True}

        chars = script_data.get("characters", [])
        sets = script_data.get("settings", [])
        is_zh = any("\u4e00" <= c <= "\u9fff" for c in script_data.get("title", ""))
        shot_prompt_tpl = _get_shot_prompt("zh" if is_zh else "en")
        
        self._report_progress("分镜", f"开始生成 {len(episodes_to_proc)} 集缺失的分镜...", 5)
        
        async def proc_ep(ep):
            # ... (保持原本处理逻辑不变)
            ep_n = ep.get("episode_number", 1)
            # ... (以下为原本 proc_ep 逻辑)
            ep_t = ep.get("act_title", f"第{ep_n}集")
            ep_c = ep.get("content", "")
            
            # 清洗剧本内容：按行拆分，用于提示说明“一行一分镜”
            lines = [line.strip() for line in ep_c.split('\n') if line.strip()]
            script_text_with_lines = "\n".join([f"L{idx+1}: {line}" for idx, line in enumerate(lines)])

            prompt = shot_prompt_tpl.format(
                act_number=ep_n, 
                act_title=ep_t, 
                script_text=script_text_with_lines, 
                asset_characters=json.dumps(chars, ensure_ascii=False), 
                asset_settings=json.dumps(sets, ensure_ascii=False),
                style=style
            )
            
            llm = LLM()
            loop = asyncio.get_running_loop()
            
            # 增加重试机制 (Max 3 retries)
            max_retries = 3
            extracted = None
            raw = ""
            
            for attempt in range(max_retries):
                try:
                    self._report_progress("分镜", f"生成集数 {ep_n} (第 {attempt + 1} 次尝试)...", 5 + attempt * 2)
                    raw = await loop.run_in_executor(None, self._cancellable_query, llm, prompt, [], llm_model, False, sid, False)
                    extracted = self._extract_json_array(raw)
                    if extracted:
                        break
                    logger.warning(f"Episode {ep_n} Attempt {attempt + 1}: Failed to extract JSON array. Retrying...")
                except Exception as e:
                    logger.error(f"Episode {ep_n} Attempt {attempt + 1}: LLM query error: {str(e)}")
                    if attempt == max_retries - 1: raise e
            
            if not extracted:
                logger.error(f"LLM output failed to parse as JSON for Episode {ep_n} after {max_retries} attempts. Raw: {raw}")
                # 抛出异常以触发 orchestrator 的错误状态处理逻辑
                raise Exception(f"第 {ep_n} 集分镜生成失败：模型输出无法解析")
                
            valid_segments = []
            for i, seg in enumerate(extracted, 1):
                if not isinstance(seg, dict): continue
                
                shots = seg.get("shots", [])
                valid_shots = []
                calc_total_duration = 0
                for s in shots:
                    if not isinstance(s, dict): continue
                    dur = s.get("duration", 5)
                    calc_total_duration += dur
                    valid_shots.append({
                        "shot_number": s.get("shot_number", len(valid_shots) + 1),
                        "shot_type": s.get("shot_type", "中景"),
                        "duration": dur,
                        "content": s.get("content", "")
                    })
                
                valid_segments.append({
                    "segment_id": seg.get("segment_id", f"seg_{ep_n:02d}_{i:02d}"),
                    "segment_number": seg.get("segment_number", len(valid_segments) + 1),
                    "total_duration": seg.get("total_duration", calc_total_duration),
                    "location": seg.get("location", ""),
                    "characters": seg.get("characters", []),
                    "shots": valid_shots,
                    "episode_number": ep_n
                })

            return {
                "episode_number": ep_n,
                "episode_title": ep_t,
                "segments": valid_segments
            }

        # 核心：支持流式保存增量产物，让前端能看到实时进度
        updated_ep_map = {e["episode_number"]: e for e in existing_story_eps}
        
        # 立即先保存一次，确保已有的 episodes 在进入 running 状态后依然可见
        session_data.setdefault("artifacts", {})["storyboard"] = {
            "session_id": sid, 
            "episodes": sorted(updated_ep_map.values(), key=lambda x: x["episode_number"]),
            "created_at": datetime.now().isoformat()
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        # 报告一次进度，带上 asset_complete 强制前端从磁盘刷新一次初步数据
        self._report_progress("分镜设计", "准备生成分镜...", 10, {"asset_complete": True})

        results_queue = [proc_ep(ep) for ep in episodes_to_proc]

        for coro in asyncio.as_completed(results_queue):
            res = await coro
            updated_ep_map[res["episode_number"]] = res
            
            # 每完成一集分镜，立即持久化并触发增量同步
            temp_eps = sorted(updated_ep_map.values(), key=lambda x: x["episode_number"])
            session_data.setdefault("artifacts", {})["storyboard"] = {
                "session_id": sid, 
                "episodes": temp_eps, 
                "created_at": datetime.now().isoformat()
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # 报告带有 asset_complete 的进度，强制编排器刷新前端数据
            self._report_progress("分镜设计", f"集数 {res['episode_number']} 分镜已生成", 50, {"asset_complete": True})

        final_all_episodes = sorted(updated_ep_map.values(), key=lambda x: x["episode_number"])
        
        # 核心：将结果持久化到 artifacts.storyboard
        session_data.setdefault("artifacts", {})["storyboard"] = {
            "session_id": sid, 
            "episodes": final_all_episodes, 
            "created_at": datetime.now().isoformat()
        }
        
        session_data["updated_at"] = datetime.now().timestamp()

        with open(session_file, "w", encoding="utf-8") as f: 
            json.dump(session_data, f, indent=2, ensure_ascii=False)
            
        self._report_progress("分镜", "完成", 100)
        return {"payload": {"session_id": sid, "episodes": final_all_episodes}, "stage_completed": True}
