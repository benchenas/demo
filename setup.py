import setuptools

setuptools.setup(
    name="demo",
    packages=setuptools.find_packages(),
    include_package_data=True,
    version="1.0.1",
    author="xiaoyang",
    author_email="yangx9810@163.com",
    description="A method of benchmark",
    long_description='long_description',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'paramiko',  'numpy',  'multiprocess',
        'torch==1.6.0', 'torchvision==0.7.0',
        'redis'
    ],
)
