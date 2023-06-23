import argparse
import os
import requests


def install_dependency(dependency):
  """Installs a dependency using pip."""
  command = "pip install {}".format(dependency)
  os.system(command)


def download_dependency(dependency):
  """Downloads a dependency from PyPI."""
  url = "https://pypi.org/simple/{}/".format(dependency)
  response = requests.get(url)
  if response.status_code == 200:
    filename = response.content
    with open(dependency, "wb") as f:
      f.write(filename)


def main():
  dependencies = ["tensorflow", "OpenCv", "deepface", "argparse"]

  for dependency in dependencies:
    if not os.path.exists(dependency):
      download_dependency(dependency)
    install_dependency(dependency)


if __name__ == "__main__":
  main()
