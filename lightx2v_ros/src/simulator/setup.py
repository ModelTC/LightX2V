from setuptools import find_packages, setup

package_name = "simulator"

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
    description="Simulation environment nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "libero_node = simulator.libero_node.main:main",
        ],
    },
)
