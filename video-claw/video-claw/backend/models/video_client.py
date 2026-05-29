"""
统一视频生成客户端
根据 model 名称自动路由到对应后端：
  - wan*      → DashscopeVideoClient (DashScope VideoSynthesis)
  - kling*    → KlingVideoClient (可灵 AI)
"""

import os
import sys

models_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(models_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import logging
from typing import Optional
from config import Config

try:
    from models.video_dashscope import DashscopeVideoClient
    from models.video_kling import KlingVideoClient
    from models.video_seedance import SeedanceVideoClient
except ImportError:
    from video_dashscope import DashscopeVideoClient
    from video_kling import KlingVideoClient
    from video_seedance import SeedanceVideoClient

logger = logging.getLogger(__name__)


class VideoClient:
    """
    统一视频生成客户端
    参照 ImageClient 模式，按模型名路由到不同后端
    """

    def __init__(
        self,
        dashscope_api_key: Optional[str] = None,
        dashscope_base_url: Optional[str] = None,
        kling_access_key: Optional[str] = None,
        kling_secret_key: Optional[str] = None,
        kling_base_url: Optional[str] = None,
        ark_api_key: Optional[str] = None,
        ark_base_url: Optional[str] = None,
    ):
        # 万象客户端
        self.Dashscope_client = DashscopeVideoClient(
            api_key=dashscope_api_key or Config.DASHSCOPE_API_KEY,
            base_url=dashscope_base_url or Config.DASHSCOPE_BASE_URL,
        )

        # 可灵客户端
        self.kling_client = KlingVideoClient(
            access_key=kling_access_key or Config.KLING_ACCESS_KEY,
            secret_key=kling_secret_key or Config.KLING_SECRET_KEY,
            base_url=kling_base_url or Config.KLING_BASE_URL,
        )

        # Seedance 客户端
        self.seedance_client = SeedanceVideoClient(
            api_key=ark_api_key or Config.ARK_API_KEY,
            base_url=ark_base_url or Config.ARK_BASE_URL,
        )

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str = "wan2.7-i2v",
        duration: int = 5,
        shot_type: str = "multi",
        sound: str = "",
        video_ratio: str = "16:9",
        resolution: Optional[str] = None,
        last_image_path: Optional[str] = None,
        first_clip_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        reference_image_paths: Optional[list[str]] = None,
        reference_video_paths: Optional[list[str]] = None,
        reference_audio_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        seed: Optional[int] = None,
        mode: str = "pro",
        cfg_scale: float = 0.5,
        generate_audio: Optional[bool] = None,
        audio: Optional[bool] = None,
    ) -> str:
        """
        生成视频

        Args:
            prompt: 视频描述提示词
            image_path: 输入图片本地路径；DashScope wan2.7 视频续写可为空并使用 first_clip_path
            save_path: 输出视频保存路径
            model: 模型名，决定使用哪个后端
            duration: 视频时长（秒）
            shot_type: 镜头类型 "single" / "multi"

        Returns:
            video_url: 远端视频 URL

        Raises:
            FileNotFoundError: 输入图片不存在
            RuntimeError: 生成或下载失败
        """
        if not model:
            model = "wan2.7-i2v"

        # 确保 duration 是整数,视频模型通常要求整数秒
        duration = int(duration)

        if Config.PRINT_MODEL_INPUT:
            lines = [
                "---- VIDEO GENERATION REQUEST ----",
                f"Prompt: {prompt}",
                "Image: [Base64图片]" if image_path and str(image_path).startswith("data:") else f"Image: {image_path}",
                f"Model: {model}",
                f"Duration: {duration}s",
                f"Shot Type: {shot_type}",
                f"Video Ratio: {video_ratio}",
            ]
            if resolution:
                lines.append(f"Resolution: {resolution}")
            if last_image_path:
                lines.append(f"Last Image: {last_image_path}")
            if first_clip_path:
                lines.append(f"First Clip: {first_clip_path}")
            if reference_image_path:
                lines.append(f"Reference Image: {reference_image_path}")
            if reference_image_paths:
                lines.append(f"Reference Images: {reference_image_paths}")
            if reference_video_paths:
                lines.append(f"Reference Videos: {reference_video_paths}")
            if reference_audio_path:
                lines.append(f"Reference Audio: {reference_audio_path}")
            if audio_path:
                lines.append(f"Audio: {audio_path}")
            if negative_prompt:
                lines.append(f"Negative Prompt: {negative_prompt}")
            if sound:
                lines.append(f"Sound: {sound}")
            lines.extend([
                f"Save: {save_path}",
                "-" * 30,
            ])
            logger.info("\n%s", "\n".join(lines))

        model_lower = model.lower()

        if "kling" in model_lower:
            return self._generate_kling(
                prompt,
                image_path,
                save_path,
                model,
                duration,
                sound,
                mode,
                cfg_scale,
                negative_prompt or "",
            )
        elif "seedance" in model_lower:
            return self._generate_seedance(
                prompt,
                image_path,
                save_path,
                model,
                duration,
                video_ratio,
                resolution,
                seed,
                watermark,
                generate_audio,
            )
        elif "wan" in model_lower or "happyhorse" in model_lower:
            return self._generate_wan(
                prompt,
                image_path,
                save_path,
                model,
                duration,
                shot_type,
                video_ratio,
                last_image_path,
                first_clip_path,
                reference_image_path,
                reference_image_paths,
                reference_video_paths,
                reference_audio_path,
                audio_path,
                negative_prompt,
                resolution,
                prompt_extend,
                watermark if watermark is not None else False,
                seed,
                audio,
            )
        else:
            raise ValueError(f"未知的视频生成模型: {model}")

    def _generate_wan(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str,
        duration: int,
        shot_type: str,
        video_ratio: str,
        last_image_path: Optional[str],
        first_clip_path: Optional[str],
        reference_image_path: Optional[str],
        reference_image_paths: Optional[list[str]],
        reference_video_paths: Optional[list[str]],
        reference_audio_path: Optional[str],
        audio_path: Optional[str],
        negative_prompt: Optional[str],
        resolution: Optional[str],
        prompt_extend: Optional[bool],
        watermark: bool,
        seed: Optional[int],
        audio: Optional[bool],
    ) -> str:
        """通过万象模型生成视频"""
        # 设置 Wan 和 Happyhorse 系列视频生成的默认分辨率为 720P
        resolution = resolution or "720P"
        
        logger.info("VideoClient routed to Wan: model=%s", model)
        return self.Dashscope_client.generate_video(
            prompt=prompt,
            image_path=image_path,
            save_path=save_path,
            model=model,
            duration=duration,
            shot_type=shot_type,
            video_ratio=video_ratio,
            last_image_path=last_image_path,
            first_clip_path=first_clip_path,
            reference_image_path=reference_image_path,
            reference_image_paths=reference_image_paths,
            reference_video_paths=reference_video_paths,
            reference_audio_path=reference_audio_path,
            audio_path=audio_path,
            negative_prompt=negative_prompt,
            resolution=resolution,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            audio=audio,
        )

    def _generate_kling(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str,
        duration: int = 5,
        sound: str = "",
        mode: str = "pro",
        cfg_scale: float = 0.5,
        negative_prompt: str = "",
    ) -> str:
        """通过可灵模型生成视频"""
        logger.info("VideoClient routed to Kling: model=%s", model)
        return self.kling_client.generate_video(
            prompt=prompt,
            image_path=image_path,
            save_path=save_path,
            model=model,
            duration=duration,
            sound=sound,
            mode=mode,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
        )

    def _generate_seedance(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str,
        duration: int = 5,
        video_ratio: str = "16:9",
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        watermark: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
    ) -> str:
        """通过 Seedance 模型生成视频"""
        logger.info("VideoClient routed to Seedance: model=%s", model)
        return self.seedance_client.generate_video(
            prompt=prompt,
            image_path=image_path,
            save_path=save_path,
            model=model,
            duration=duration,
            ratio=video_ratio,
            resolution=resolution or "720p",
            seed=seed,
            watermark=watermark,
            generate_audio=generate_audio,
        )
