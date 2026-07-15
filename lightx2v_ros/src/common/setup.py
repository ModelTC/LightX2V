from setuptools import find_packages, setup

package_name = "common"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Shared environment contract for LightX2V ROS nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [],
    },
)
