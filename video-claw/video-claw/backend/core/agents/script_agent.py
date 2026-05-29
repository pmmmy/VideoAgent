# -*- coding: utf-8 -*-
"""
阶段1: 编剧智能体 (直出一遍过版本)
"""

import os
import re
import json
import asyncio
import logging
from functools import partial
from datetime import datetime, timezone
from typing import Any, Optional, Dict

from prompts.loader import load_prompt_with_fallback
from .base_agent import AgentInterface

logger = logging.getLogger(__name__)

def _get_script_prompt(name: str, lang: str = "zh") -> str:
    return load_prompt_with_fallback("script", name, lang, "zh")

class ScriptWriterAgent(AgentInterface):
    def __init__(self):
        super().__init__(name="ScriptWriter")

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Any]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试匹配第一个 { 或 [ 到底部对应的 } 或 ]
        start_obj = text.find('{')
        start_arr = text.find('[')
        
        # 确定起始位置
        if start_obj == -1 and start_arr == -1:
            return None
        
        start = start_obj if (start_obj != -1 and (start_arr == -1 or start_obj < start_arr)) else start_arr
        end_char = '}' if start == start_obj else ']'
        end = text.rfind(end_char)

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def _gen_id(self, prefix: str = "char") -> str:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:6]}"

    def _save_result(self, json_data: dict, sid: str, is_zh: bool):
        from config import settings as app_settings
        os.makedirs(os.path.join(app_settings.RESULT_DIR, 'script'), exist_ok=True)
        out_path = os.path.join(app_settings.RESULT_DIR, 'script', f'{sid}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[ScriptWriter] script saved to {out_path}")

    def _save_progress(self, sid: str, phase: str, data: dict):
        pass

    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        if intervention and "modified_script" in intervention:
            modified = intervention["modified_script"]
            sid = input_data.get("session_id", "")
            if isinstance(modified, str):
                modified = self._extract_json_from_text(modified) or {}
            is_zh = any('\u4e00' <= c <= '\u9fff' for c in modified.get("title", ""))
            modified["session_id"] = sid
            # 【优化】移除手动调用 self._save_result，依靠 Orchestrator 自动保存
            return {"payload": modified, "requires_intervention": False, "stage_completed": True}

        # 处理确认续写或删除续写的结果，更新script_genenration和character_design数据结构，并保存最终结果
        if intervention and intervention.get("action") in ["confirm_continue", "delete_continue"]:
            import copy
            final_data = copy.deepcopy(input_data)
            sid = final_data.get("session_id", "")
            
            if intervention.get("action") == "confirm_continue":
                new_chars = final_data.get("new_characters", [])
                new_settings = final_data.get("new_settings", [])
                new_ep_list = final_data.get("new_episodes", [])
                
                # 更新第一阶段剧本数据 (内存)
                final_data.setdefault("episodes", []).extend(new_ep_list)
                final_data.setdefault("characters", []).extend(new_chars)
                final_data.setdefault("settings", []).extend(new_settings)

                # 创建一个包含增量信息的返回结果，供 Orchestrator 钩子使用
                result_payload = copy.deepcopy(final_data)
                result_payload["new_characters"] = new_chars
                result_payload["new_settings"] = new_settings
                result_payload["new_episodes"] = new_ep_list

                logger.info(f"[ScriptWriter] Confirmed continuation. Providing incremental data to Orchestrator.")
                return {"payload": result_payload, "requires_intervention": False, "stage_completed": True}

            # 处理 delete_continue 的情况，直接丢弃新增内容，保持原有剧本数据不变
            for key in ["new_episodes", "new_characters", "new_settings", "sequel_idea"]:
                final_data.pop(key, None)
            return {"payload": final_data, "requires_intervention": False, "stage_completed": True}
        # ---------------------------------------------------- #

        async def run_smart_continue():
            import copy
            sid = input_data.get("session_id", "")
            llm_model = self._require_input(input_data, "llm_model")
            web_search = input_data.get("web_search", False)
            episodes_to_add = intervention.get("episodes_to_add", 1)
            sequel_idea = intervention.get("sequel_idea", "").strip()

            from config import settings as app_settings
            from models.llm_client import LLM
            llm = LLM()

            def _log_progress(pct, msg):
                self._report_progress("智能续写", msg, pct)
                logger.info(f"[{pct}%] {msg}")

            loop = asyncio.get_running_loop()
            
            existing_episodes_text = json.dumps(input_data.get("episodes", []), ensure_ascii=False)
            existing_chars_text = json.dumps(input_data.get("characters", []), ensure_ascii=False)
            existing_settings_text = json.dumps(input_data.get("settings", []), ensure_ascii=False)
            
            last_episode_num = 0
            if input_data.get("episodes"):
                last_episode_num = input_data["episodes"][-1].get("episode_number", len(input_data["episodes"]))

            if not sequel_idea:
                _log_progress(10, "生成续写灵感...")
                idea_prompt = f"根据以下已有的剧集内容，在100字内，提供一个后续{episodes_to_add}集的简短续写灵感(主线方向): {existing_episodes_text}"
                sequel_idea = await loop.run_in_executor(None, self._cancellable_query, llm, idea_prompt, [], llm_model, True, sid, web_search)
                sequel_idea = sequel_idea.strip()

            _log_progress(30, "正在生成续写剧本文本...")
            prompt_name = "smart_continue_script"
            prompt = _get_script_prompt(prompt_name, "zh").format(
                episodes_text=existing_episodes_text,
                chars_text=existing_chars_text,
                settings_text=existing_settings_text,
                episodes_to_add=episodes_to_add,
                sequel_idea=sequel_idea,
                start_episode_num=last_episode_num + 1
            )
            
            _log_progress(45, "正在生成续写初稿...")
            sequel_script_text = await loop.run_in_executor(None, self._cancellable_query, llm, prompt, [], llm_model, True, sid, web_search)

            _log_progress(50, "正在进行台词评估...")
            eval_dialogue_prompt = _get_script_prompt("eval_dialogue", "zh" if is_zh else "en").format(script_text=sequel_script_text)
            dialogue_critique = await loop.run_in_executor(None, self._cancellable_query, llm, eval_dialogue_prompt, [], llm_model, True, sid, web_search)
            
            _log_progress(55, "正在进行情节评估...")
            eval_plot_prompt = _get_script_prompt("eval_plot", "zh" if is_zh else "en").format(script_text=sequel_script_text)
            plot_critique = await loop.run_in_executor(None, self._cancellable_query, llm, eval_plot_prompt, [], llm_model, True, sid, web_search)
            
            _log_progress(58, "正在根据评估意见优化续写内容...")
            revise_prompt = _get_script_prompt("revise_script", "zh" if is_zh else "en").format(
                script_text=sequel_script_text, 
                dialogue_critique=dialogue_critique, 
                plot_critique=plot_critique
            )
            sequel_script_text = await loop.run_in_executor(None, self._cancellable_query, llm, revise_prompt, [], llm_model, True, sid, web_search)

            _log_progress(60, "提取新增人物/场景...")
            meta_prompt = _get_script_prompt("meta_extract_sequel", "zh").format(
                existing_chars=existing_chars_text,
                existing_settings=existing_settings_text,
                sequel_script=sequel_script_text
            )
            meta_raw = await loop.run_in_executor(None, self._cancellable_query, llm, meta_prompt, [], llm_model, True, sid, web_search)
            meta_res = self._extract_json_from_text(meta_raw)
            meta_data = meta_res if isinstance(meta_res, dict) else {}

            new_chars = meta_data.get("new_characters", [])
            new_settings = meta_data.get("new_settings", [])
            for c in new_chars:
                c["character_id"] = self._gen_id("char")
            for s in new_settings:
                s["setting_id"] = self._gen_id("set")

            _log_progress(80, "结构化续写集数据...")
            extract_prompt = _get_script_prompt("act_extract_sequel", "zh").format(
                sequel_script=sequel_script_text,
                start_episode_num=last_episode_num + 1,
                episodes_to_add=episodes_to_add
            )
            
            new_episodes = []
            max_retries = 3
            raw_acts = ""
            for attempt in range(max_retries):
                raw_acts = await loop.run_in_executor(None, self._cancellable_query, llm, extract_prompt, [], llm_model, True, sid, web_search)
                parsed_acts = self._extract_json_from_text(raw_acts)
                
                new_episodes.clear()
                if isinstance(parsed_acts, list):
                    for act in parsed_acts:
                        if isinstance(act, dict):
                            new_episodes.append({
                                "episode_number": act.get("episode_number"),
                                "act_title": act.get("act_title") or f"第{act.get('episode_number')}集",
                                "content": act.get("content", "")
                            })
                elif isinstance(parsed_acts, dict):
                    act_list = parsed_acts.get("new_episodes") or parsed_acts.get("episodes") or list(parsed_acts.values())[0]
                    if isinstance(act_list, list):
                        for act in act_list:
                            if isinstance(act, dict):
                                new_episodes.append({
                                    "episode_number": act.get("episode_number"),
                                    "act_title": act.get("act_title") or f"第{act.get('episode_number')}集",
                                    "content": act.get("content", "")
                                })
                
                if new_episodes:
                    break
                logger.warning(f"[ScriptWriter] Extraction failed on attempt {attempt+1}, retrying...")
                _log_progress(85, f"数据解析失败，自动进行第 {attempt+1} 次重试...")

            # 最终兜底：如果重试多次依然失败，直接将返回的文本全塞进一集里
            if not new_episodes and sequel_script_text:
                logger.error(f"[ScriptWriter] All {max_retries} attempts to parse new episodes failed.")
                new_episodes.append({
                    "episode_number": last_episode_num + 1,
                    "act_title": f"第{last_episode_num + 1}集 续集",
                    "content": sequel_script_text.strip()
                })

            final_data = copy.deepcopy(input_data)
            final_data["new_episodes"] = new_episodes
            final_data["new_characters"] = new_chars
            final_data["new_settings"] = new_settings
            final_data["sequel_idea"] = sequel_idea
            
            is_zh = any('\u4e00' <= c <= '\u9fff' for c in final_data.get("title", "Generated Script"))
            self._save_result(final_data, sid, is_zh)
            _log_progress(100, "智能续写完成")
            return final_data

        if intervention and intervention.get("action") == "smart_continue":
            result = await run_smart_continue()
            # 设置 requires_intervention=True 以触发表单确认按钮
            return {"payload": result, "requires_intervention": True, "stage_completed": False}

        async def run_logic():
            idea = input_data.get("idea", "")
            sid = input_data.get("session_id", "")
            style = input_data.get("style", "anime")
            llm_model = self._require_input(input_data, "llm_model")
            web_search = input_data.get("web_search", False)
            episodes = input_data.get("episodes")
            if episodes is None:
                logger.warning("[ScriptWriter] episodes missing from input_data; falling back to 4. session=%s", sid)
                episodes = 4
            try:
                episodes = max(1, int(episodes))
            except (TypeError, ValueError):
                logger.warning("[ScriptWriter] invalid episodes=%r; falling back to 4. session=%s", episodes, sid)
                episodes = 4
            is_zh = any('\u4e00' <= c <= '\u9fff' for c in idea)

            from config import settings as app_settings
            from models.llm_client import LLM
            os.makedirs(app_settings.TEMP_DIR, exist_ok=True)
            llm = LLM()

            def _log_progress(pct, msg):
                self._report_progress("剧本生成", msg, pct)
                logger.info(f"[{pct}%] {msg}")

            # 1. Generate full script
            _log_progress(10, "正在生成完整剧本文本初稿...")
            prompt = _get_script_prompt("generate_script", "zh" if is_zh else "en").format(idea=idea, style=style, episodes=episodes)

            loop = asyncio.get_running_loop()
            full_script_text = await loop.run_in_executor(None, self._cancellable_query, llm, prompt, [], llm_model, True, sid, web_search)
            logger.info(f"[ScriptWriter] Initial script generated ({len(full_script_text)} chars)")
            
            _log_progress(20, "正在进行台词评估...")
            eval_dialogue_prompt = _get_script_prompt("eval_dialogue", "zh" if is_zh else "en").format(script_text=full_script_text)
            dialogue_critique = await loop.run_in_executor(None, self._cancellable_query, llm, eval_dialogue_prompt, [], llm_model, True, sid, web_search)
            
            _log_progress(30, "正在进行情节评估...")
            eval_plot_prompt = _get_script_prompt("eval_plot", "zh" if is_zh else "en").format(script_text=full_script_text)
            plot_critique = await loop.run_in_executor(None, self._cancellable_query, llm, eval_plot_prompt, [], llm_model, True, sid, web_search)
            
            _log_progress(40, "正在根据评估意见优化剧本...")
            revise_prompt = _get_script_prompt("revise_script", "zh" if is_zh else "en").format(
                script_text=full_script_text, 
                dialogue_critique=dialogue_critique, 
                plot_critique=plot_critique
            ) + prompt  # 将原始生成提示词追加到优化提示词末尾，提供更多上下文信息帮助优化
            full_script_text = await loop.run_in_executor(None, self._cancellable_query, llm, revise_prompt, [], llm_model, True, sid, web_search)
            logger.info(f"[ScriptWriter] Final script generated ({len(full_script_text)} chars)")

            _log_progress(60, "最终剧本生成完成，正在提取人物/场景信息...")

            # 2. Extract meta data -> total_episodes, characters, settings
            meta_prompt = _get_script_prompt("meta_extract", "zh" if is_zh else "en").format(script_text=full_script_text, outline=full_script_text)
            meta_raw = await loop.run_in_executor(None, self._cancellable_query, llm, meta_prompt, [], llm_model, True, sid, web_search)
            meta_res = self._extract_json_from_text(meta_raw)
            meta_data = meta_res if isinstance(meta_res, dict) else {}

            all_characters = meta_data.get("characters", [])
            all_settings = meta_data.get("settings", [])
            for c in all_characters:
                c["character_id"] = c.get("character_id") or self._gen_id("char")
            for s in all_settings:
                s["setting_id"] = s.get("setting_id") or self._gen_id("set")

            asset_chars_str = json.dumps([{"name": c.get("name"), "description": c.get("description"), "role": c.get("role")} for c in all_characters], ensure_ascii=False)
            asset_sets_str = json.dumps([{"name": s.get("name"), "description": s.get("description")} for s in all_settings], ensure_ascii=False)

            # 3. 解析各集数据 - 针对新版数组输出格式进行优化
            _log_progress(80, "开始结构化全集数据...")
            
            extract_prompt = _get_script_prompt("act_extract", "zh" if is_zh else "en").format(
                script_text=full_script_text
            )
            
            raw_acts = await loop.run_in_executor(None, self._cancellable_query, llm, extract_prompt, [], llm_model, True, sid, web_search)
            parsed_acts = self._extract_json_from_text(raw_acts)
            
            all_episodes = []
            if isinstance(parsed_acts, list):
                for act in parsed_acts:
                    if isinstance(act, dict):
                        all_episodes.append({
                            "episode_number": act.get("episode_number"),
                            "act_title": act.get("act_title") or f"第{act.get('episode_number')}集",
                            "content": act.get("content", "")
                        })

            if not all_episodes:
                logger.error(f"[ScriptWriter] Failed to parse episodes from LLM output. Raw: {raw_acts[:200]}...")
            
            final_json = {
                "project_id": f"proj_{sid}",
                "session_id": sid,
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "meta": {
                    "generation_model": llm_model,
                    "generation_prompt": idea,
                    "original_text": full_script_text
                },
                "title": meta_data.get("title", "Generated Script"),
                "logline": meta_data.get("logline", ""),
                "genre": meta_data.get("genre", []),
                "mood": meta_data.get("mood", ""),
                "characters": all_characters,
                "settings": all_settings,
                "episodes": all_episodes
            }
            
            self._save_result(final_json, sid, is_zh)
            _log_progress(100, "剧本结构化解析完成！")
            return final_json

        result = await run_logic()
        return {"payload": result, "requires_intervention": False, "stage_completed": True}
