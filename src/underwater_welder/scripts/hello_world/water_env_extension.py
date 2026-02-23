# water_env_extension.py

import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .water_env import Water_Env

class WaterEnvExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Water Environment"
        self.category = "rokey" # 사용자 지정 카테고리

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Underwater Refined Physics",
            "doc_link": "",
            "overview": "정밀 샘플링 기반 부력, 항력 및 정수압 모델 시뮬레이션입니다.",
            "sample": Water_Env(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )
        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        return