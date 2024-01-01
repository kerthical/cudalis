use std::env;
use std::process::exit;
use std::time::Duration;

use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, LogOutput, RemoveContainerOptions, StartContainerOptions};
use bollard::Docker;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::image::{CommitContainerOptions, CreateImageOptions};
use clap::{Arg, ArgAction, Command};
use futures_util::stream::StreamExt;

#[derive(Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Hash)]
struct PyTorchVersion {
    name: String,
    torch: String,
    python: String,
    accelerator: String,
    os: String,
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
            torch: version,
            python,
            os,
            accelerator,
            href: href.to_string(),
        })
    }

    fn get_python_semantic_version(&self) -> String {
        let python_version = self.python.replace("cp", "");
        let major = python_version.chars().nth(0).unwrap();
        let minor = python_version.chars().nth(1).unwrap();
        format!("{}.{}", major, minor)
    }

    fn get_accelerator_semantic_version(&self) -> String {
        if self.accelerator == "cpu" {
            "cpu".to_string()
        } else {
            let cuda_version = self.accelerator.replace("cu", "");
            let major = &cuda_version[0..2];
            let minor = &cuda_version[2..3];
            format!("{}.{}", major, minor)
        }
    }
}

#[tokio::main]
async fn main() {
    let command = Command::new("cudalis")
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .arg(Arg::new("python").short('p').long("python").value_name("VERSION"))
        .arg(Arg::new("torch").short('t').long("torch").value_name("VERSION"))
        .arg(Arg::new("cuda").short('c').long("cuda").value_name("VERSION"))
        .arg(Arg::new("verbose").short('v').long("verbose").action(ArgAction::SetTrue));

    let matches = command.get_matches();

    let python_version = matches.get_one::<String>("python").map(|v| format_python_version(v));
    let torch_version = matches.get_one::<String>("torch").map(|v| v.to_string());
    let mut cuda_version = matches.get_one::<String>("cuda").map(|v| format_cuda_version(v));
    let verbose = matches.get_flag("verbose");

    if cuda_version.is_none() && env::consts::OS == "macos" {
        cuda_version = Some("cpu".to_string());
    }

    println!("[+] Resolving versions with python {}, torch {}, and cuda {}...", python_version.as_ref().unwrap_or(&"latest".to_string()), torch_version.as_ref().unwrap_or(&"latest".to_string()), cuda_version.as_ref().unwrap_or(&"latest".to_string()));

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
    versions = filter_versions_by_specified_version(versions, torch_version, |v| &v.torch);
    versions = filter_versions_by_specified_version(versions, cuda_version, |v| &v.accelerator);

    versions.sort_by(|a, b| a.torch.cmp(&b.torch));

    if versions.is_empty() {
        eprintln!("No versions found with the specified constraints");
        exit(1);
    }

    let versions = versions.last().unwrap();

    let mut docker = Docker::connect_with_local_defaults().unwrap();
    docker.set_timeout(Duration::from_secs(3600));
    let base_image_tag = if versions.accelerator == "cpu" {
        "ubuntu:22.04".to_string()
    } else {
        let result1: serde_json::Value = reqwest::get(&format!(
            "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100&name={}",
            versions.get_accelerator_semantic_version()
        ))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let tag = result1["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| {
                let tag = t["name"].as_str().unwrap().to_string();

                tag.starts_with(&versions.get_accelerator_semantic_version()) && tag.contains("ubuntu") && tag.contains("devel")
            })
            .max_by(|a, b| {
                let a_tag = a["name"].as_str().unwrap().to_string();
                let b_tag = b["name"].as_str().unwrap().to_string();

                a_tag.cmp(&b_tag)
            });

        if tag.is_none() {
            eprintln!("No CUDA image found for version {}", versions.get_accelerator_semantic_version());
            exit(1);
        }

        format!("nvidia/cuda:{}", tag.unwrap()["name"].as_str().unwrap().to_string())
    };

    println!("[+] Using resolved versions: python {}, torch {}, cuda {}, and tag {}", versions.get_python_semantic_version(), versions.torch, versions.get_accelerator_semantic_version(), base_image_tag);

    let stream = docker.create_image(
        Some(CreateImageOptions {
            from_image: base_image_tag.clone(),
            ..Default::default()
        }),
        None,
        None,
    );

    let mut stream = stream.map(|result1| match result1 {
        Ok(_) => {}
        Err(e) => {
            eprintln!("[!] Error pulling the base image: {}", e);
            exit(1);
        }
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
            }
        }
    }

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

    if base_image_tag.contains("20.04") {
        execute_commands(
            &docker,
            "cudalis_setup",
            vec![
                "echo 'deb http://archive.ubuntu.com/ubuntu bionic main universe' >> /etc/apt/sources.list",
                "echo 'deb http://archive.ubuntu.com/ubuntu bionic-security main universe' >> /etc/apt/sources.list",
                "echo 'deb http://archive.ubuntu.com/ubuntu bionic-updates main universe' >> /etc/apt/sources.list",
            ],
            verbose,
        ).await;
    }

    execute_commands(
        &docker,
        "cudalis_setup",
        vec![
            "sed -i.bak -r 's@http://(jp\\.)?archive\\.ubuntu\\.com/ubuntu/?@https://ftp.udx.icscoe.jp/Linux/ubuntu/@g' /etc/apt/sources.list",
            &format!(
                "export DEBIAN_FRONTEND=noninteractive && apt update -y && apt upgrade -y && apt install -y python{} python{}-dev python{}-distutils",
                versions.get_python_semantic_version(),
                versions.get_python_semantic_version(),
                versions.get_python_semantic_version(),
            ),
            "pip install torch=={} torchvision torchaudio -f https://download.pytorch.org/whl/{}",
        ],
        verbose,
    ).await;

    let image_name = format!(
        "cudalis:{}-{}-{}",
        versions.get_python_semantic_version(),
        versions.torch,
        versions.get_accelerator_semantic_version()
    );
    docker
        .commit_container(
            CommitContainerOptions {
                container: "cudalis_setup",
                repo: &image_name,
                ..Default::default()
            },
            Config {
                image: Some(base_image_tag.clone()),
                healthcheck: None,
                ..Default::default()
            },
        )
        .await
        .unwrap();

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
    docker.remove_image(base_image_tag.as_str(), None, None).await.unwrap();

    println!("[+] Done. You can now use the image with tag: {}", image_name);
}

async fn execute_commands(docker: &Docker, container_id: &str, commands: Vec<&str>, verbose: bool) {
    for command in commands {
        execute_command(docker, container_id, command, verbose).await;
    }
}

async fn execute_command(docker: &Docker, container_id: &str, command: &str, verbose: bool) {
    println!("[+] {}", command);

    let id = docker
        .create_exec(
            container_id,
            CreateExecOptions {
                cmd: Some(vec!["bash", "-c", command]),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                tty: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .id;

    let stream = docker
        .start_exec(
            &id,
            Some(StartExecOptions {
                detach: false,
                ..Default::default()
            }),
        )
        .await;

    if let Ok(StartExecResults::Attached { mut output, input: _ }) = stream {
        while let Some(result) = output.next().await {
            match result {
                Ok(LogOutput::StdOut { message }) => {
                    if verbose {
                        print!("{}", String::from_utf8_lossy(&message));
                    }
                }
                Ok(LogOutput::StdErr { message }) => {
                    eprint!("{}", String::from_utf8_lossy(&message));
                }
                _ => {}
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
            .filter(|v| version_extractor(v).contains(&specified_version))
            .collect()
    } else if let Some(latest_version) = versions
        .iter()
        .max_by(|a, b| version_extractor(a).cmp(version_extractor(b)))
        .map(|v| version_extractor(v).clone())
    {
        versions
            .into_iter()
            .filter(|v| version_extractor(v).contains(&latest_version))
            .collect()
    } else {
        Vec::new()
    }
}
