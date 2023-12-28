use std::env;

use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, LogOutput, RemoveContainerOptions, StartContainerOptions};
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::image::{CommitContainerOptions, CreateImageOptions};
use bollard::Docker;
use clap::{Arg, Command};
use futures_util::stream::StreamExt;

#[derive(Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Hash)]
struct PyTorchVersion {
    name: String,
    version: String,
    python: String,
    os: String,
    accelerator: String,
    href: String,
}

impl PyTorchVersion {
    fn parse_from_html_tag(tag: &str) -> Option<Self> {
        let parts: Vec<&str> = tag.split('"').collect();
        let href = parts.get(1)?;
        if !href.starts_with("cpu") && !href.starts_with("cu") {
            return None;
        }
        let segments: Vec<&str> = href.split('/').collect();
        let accelerator = segments[0].replace("_pypi_cudnn", "");
        let package_parts: Vec<&str> = segments[1].split('-').collect();
        let name = package_parts.first()?.to_string();
        let version = package_parts.get(1)?.split("%2B").next()?.to_string();
        let python = package_parts.get(2)?.to_string();
        let os = package_parts
            .get(4)?
            .to_string()
            .replace("win", "windows")
            .replace("macosx", "macos")
            .replace("manylinux", "linux")
            .replace("amd64", "x86_64")
            .replace("arm64", "aarch64")
            .replace(".whl", "");

        Some(PyTorchVersion {
            name,
            version,
            python,
            os,
            accelerator,
            href: href.to_string(),
        })
    }

    fn get_python_full_version(&self) -> String {
        let python_version = self.python.replace("cp", "");
        let major = python_version.chars().nth(0).unwrap();
        let minor = python_version.chars().nth(1).unwrap();
        format!("{}.{}", major, minor)
    }

    fn get_accelerator_full_version(&self) -> String {
        return if self.accelerator == "cpu" {
            "cpu".to_string()
        } else {
            let cuda_version = self.accelerator.replace("cu", "");
            let major = cuda_version.chars().nth(0).unwrap();
            let minor = cuda_version.chars().nth(1).unwrap();
            format!("{}.{}", major, minor)
        };
    }
}

#[tokio::main]
async fn main() {
    let pytorch_version = get_pytorch_version().await.unwrap();
    build_docker_image(&pytorch_version).await;
}

async fn get_pytorch_version() -> Option<PyTorchVersion> {
    let command = Command::new("cudalis")
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .arg(
            Arg::new("python")
                .short('p')
                .long("python")
                .value_name("VERSION"),
        )
        .arg(
            Arg::new("torch")
                .short('t')
                .long("torch")
                .value_name("VERSION"),
        )
        .arg(
            Arg::new("cuda")
                .short('c')
                .long("cuda")
                .value_name("VERSION"),
        );
    let matches = command.get_matches();

    let python_version = matches
        .get_one::<String>("python")
        .map(|v| format_python_version(v));
    let torch_version = matches.get_one::<String>("torch").map(|v| v.to_string());
    let mut cuda_version = matches
        .get_one::<String>("cuda")
        .map(|v| format_cuda_version(v));

    if cuda_version.is_none() && env::consts::OS == "macos" {
        cuda_version = Some("cpu".to_string());
    }

    println!("[+] Resolving the PyTorch version with the following constraints:");
    println!(
        "    Python version: {}",
        python_version.as_ref().unwrap_or(&"latest".to_string())
    );
    println!(
        "    PyTorch version: {}",
        torch_version.as_ref().unwrap_or(&"latest".to_string())
    );
    println!(
        "    CUDA version: {}",
        cuda_version.as_ref().unwrap_or(&"latest".to_string())
    );
    println!();

    let result = reqwest::get("https://download.pytorch.org/whl/torch_stable.html")
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let mut versions = result
        .split('\n')
        .filter_map(PyTorchVersion::parse_from_html_tag)
        .filter(|v| v.name == "torch")
        .collect::<Vec<_>>();

    versions = filter_versions_by_os_and_arch(versions);
    versions = filter_versions_by_specified_version(versions, python_version, |v| &v.python);
    versions = filter_versions_by_specified_version(versions, torch_version, |v| &v.version);
    versions = filter_versions_by_specified_version(versions, cuda_version, |v| &v.accelerator);

    versions.sort_by(|a, b| a.version.cmp(&b.version));

    versions.last().cloned()
}

async fn build_docker_image(pytorch_version: &PyTorchVersion) {
    println!("[+] Building the Docker image with the following constraints:");
    println!(
        "    Python version: {}",
        pytorch_version.get_python_full_version()
    );
    println!("    PyTorch version: {}", pytorch_version.version);
    println!(
        "    CUDA version: {}",
        pytorch_version.get_accelerator_full_version()
    );
    println!();

    let docker = Docker::connect_with_local_defaults().unwrap();
    let base_image_tag = if pytorch_version.accelerator == "cpu" {
        "ubuntu:22.04".to_string()
    } else {
        println!(
            "[+] Resolving docker image tag for CUDA version: {}",
            pytorch_version.get_accelerator_full_version()
        );
        let result: serde_json::Value = reqwest::get(&format!(
            "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100&name={}",
            pytorch_version.get_accelerator_full_version()
        ))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
        let tags = result["results"].as_array().unwrap();
        let tag = tags
            .iter()
            .find(|t| {
                t["name"]
                    .as_str()
                    .unwrap()
                    .starts_with(&pytorch_version.get_accelerator_full_version())
            })
            .unwrap()["name"]
            .as_str()
            .unwrap();
        format!("nvidia/cuda:{}", tag)
    };

    println!("[+] Pulling the base image: {}", base_image_tag);
    let stream = docker.create_image(
        Some(CreateImageOptions {
            from_image: base_image_tag.clone(),
            ..Default::default()
        }),
        None,
        None,
    );

    let mut stream = stream.map(|result| match result {
        Ok(_) => {}
        Err(e) => eprintln!("[!] Error pulling the base image: {}", e),
    });

    while stream.next().await.is_some() {}

    let containers = docker
        .list_containers(Some(ListContainersOptions::<String> {
            all: true,
            ..Default::default()
        }))
        .await
        .unwrap();

    for container in containers {
        if let Some(image) = container.image {
            if image == base_image_tag {
                let container_id = container.id.unwrap();
                docker
                    .remove_container(
                        container_id.clone().as_str(),
                        Some(RemoveContainerOptions {
                            force: true,
                            v: true,
                            ..Default::default()
                        }),
                    )
                    .await
                    .unwrap();
                println!("[+] Removed existing container: {}", container_id);
            }
        }
    }

    println!("[+] Creating the build container");
    docker
        .create_container(
            Some(CreateContainerOptions {
                name: "cudalis_setup",
                platform: None,
            }),
            Config {
                image: Some(base_image_tag.clone()),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                tty: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    docker
        .start_container("cudalis_setup", None::<StartContainerOptions<String>>)
        .await
        .unwrap();

    println!("[+] Installing dependencies");
    let id = docker.create_exec(
        "cudalis_setup",
        CreateExecOptions {
            cmd: Some(vec![
                "bash",
                "-c",
                "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y curl build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev git",
            ]),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            tty: Some(true),
            ..Default::default()
        },
    ).await.unwrap().id;

    let stream = docker.start_exec(&id, Some(StartExecOptions {
        detach: false,
        ..Default::default()
    })).await;

    consume_stream(stream).await;

    println!("[+] Installing pyenv and Python");
    let id = docker.create_exec(
        "cudalis_setup",
        CreateExecOptions {
            cmd: Some(vec![
                "bash",
                "-c",
                "curl https://pyenv.run | bash && echo 'export PATH=\"\\$HOME/.pyenv/bin:\\$PATH\"' >> ~/.bashrc",
            ]),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            tty: Some(true),
            ..Default::default()
        },
    ).await.unwrap().id;

    let stream = docker.start_exec(&id, None).await;

    consume_stream(stream).await;

    let id = docker
        .create_exec(
            "cudalis_setup",
            CreateExecOptions {
                cmd: Some(vec![
                    "bash",
                    "-c",
                    &format!(
                        "~/.pyenv/bin/pyenv install {} && ~/.pyenv/bin/pyenv global {}",
                        pytorch_version.get_python_full_version(),
                        pytorch_version.get_python_full_version()
                    ),
                ]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                tty: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .id;

    let stream = docker.start_exec(&id, None).await;

    consume_stream(stream).await;

    println!("[+] Installing PyTorch");
    let id = docker.create_exec(
        "cudalis_setup",
        CreateExecOptions {
            cmd: Some(vec![
                "bash",
                "-c",
                &format!("~/.pyenv/shims/pip install torch=={} -f https://download.pytorch.org/whl/{}", pytorch_version.version, pytorch_version.accelerator),
            ]),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            tty: Some(true),
            ..Default::default()
        },
    ).await.unwrap().id;

    let stream = docker.start_exec(&id, None).await;

    consume_stream(stream).await;

    let image_name = format!(
        "cudalis:{}-pytorch{}-{}",
        pytorch_version.get_python_full_version(),
        pytorch_version.version,
        pytorch_version.get_accelerator_full_version()
    );
    println!("[+] Committing the Docker image: {}", image_name);
    docker
        .commit_container(
            CommitContainerOptions {
                container: "cudalis_setup",
                repo: &image_name,
                ..Default::default()
            },
            Config {
                image: Some(base_image_tag.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    println!("[+] Stopping the build container");
    docker
        .remove_container(
            "cudalis_setup",
            Some(RemoveContainerOptions {
                force: true,
                v: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    docker
        .remove_image(base_image_tag.as_str(), None, None)
        .await
        .unwrap();

    println!("[+] Done! You can now use the image: {}", image_name);
}

async fn consume_stream(stream: Result<StartExecResults, bollard::errors::Error>) {
    if let Ok(StartExecResults::Attached { mut output, input: _ }) = stream {
        while let Some(result) = output.next().await {
            match result {
                Ok(LogOutput::StdOut { message }) => {
                    print!("{}", String::from_utf8_lossy(&message));
                }
                Ok(LogOutput::StdErr { message }) => {
                    eprint!("{}", String::from_utf8_lossy(&message));
                }
                Err(e) => {
                    eprintln!("[!] Error running the command: {}", e);
                }
                Ok(_) => {}
            }
        }
    }
}

fn format_python_version(python_version: &str) -> String {
    let python_parts = python_version.split('.').collect::<Vec<_>>();
    format!("cp{}{}", python_parts[0], python_parts[1])
}

fn format_cuda_version(cuda_version: &str) -> String {
    let cuda_parts = cuda_version.split('.').collect::<Vec<_>>();
    format!("cu{}{}", cuda_parts[0], cuda_parts[1])
}

fn filter_versions_by_os_and_arch(versions: Vec<PyTorchVersion>) -> Vec<PyTorchVersion> {
    let computer_os = env::consts::OS.to_lowercase();
    let computer_arch = env::consts::ARCH.to_lowercase();

    versions
        .into_iter()
        .filter(|v| v.os.contains(&computer_os) && v.os.contains(&computer_arch))
        .collect()
}

fn filter_versions_by_specified_version<F>(
    versions: Vec<PyTorchVersion>,
    specified_version: Option<String>,
    version_extractor: F,
) -> Vec<PyTorchVersion>
where
    F: Fn(&PyTorchVersion) -> &String,
{
    if let Some(specified_version) = specified_version {
        versions
            .into_iter()
            .filter(|v| version_extractor(v) == &specified_version)
            .collect()
    } else if let Some(latest_version) = versions
        .iter()
        .max_by(|a, b| version_extractor(a).cmp(version_extractor(b)))
        .map(|v| version_extractor(v).clone())
    {
        versions
            .into_iter()
            .filter(|v| version_extractor(v) == &latest_version)
            .collect()
    } else {
        Vec::new()
    }
}
