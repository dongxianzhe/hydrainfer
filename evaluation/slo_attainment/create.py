import argparse
import subprocess

def check_env_exists(env_name: str):
    result = subprocess.run("conda info --envs", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    envs = result.stdout.decode()
    return f"{env_name}" in envs 

def create_conda_env(env_name, vllm_version):
    try:
        print(f"creating {env_name}...")
        if not check_env_exists(env_name):
            print(f'conda env {env_name} already not exists. creating...')
            subprocess.run(f"conda create --name {env_name} python={args.python} -y", shell=True, check=True)
        else:
            print(f'conda env {env_name} already exists.')
        subprocess.run(f"conda run -n {env_name} pip install {args.package}=={vllm_version}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        print(f"creating env {env_name} failed.")
        exit(1)


def main(args: argparse.Namespace):
    for version in args.versions:
        env_name = f"{args.package}-{version.replace('.', '-')}"
        create_conda_env(env_name, version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install specified package versions and create Conda environments")
    parser.add_argument('--python', type=str, required=True, help="Specify the Python version")
    parser.add_argument('--package', type=str, required=True, help="Specify the package name")
    parser.add_argument('--versions', type=str, nargs='+', required=True, help="Specify multiple versions of the package")

    args = parser.parse_args()
    main(args)