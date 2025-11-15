import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithConfigs(build_py):
    def run(self):
        # 先执行默认构建逻辑
        super().run()
        # 把 configs 复制到构建目录下的包中
        src = os.path.join(os.getcwd(), "configs")
        dst = os.path.join(self.build_lib, "lightx2v", "configs")
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)


setup(
    cmdclass={"build_py": BuildWithConfigs},
)
