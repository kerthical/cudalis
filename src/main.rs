use std::env;

use clap::{Arg, ArgAction, Command};

use client::DockerClient;
use versions::VersionResolver;

mod client;
mod versions;

#[tokio::main]
async fn main() {
    let command = Command::new("cudalis")
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .arg(
            Arg::new("python")
                .short('p')
                .long("python")
                .value_name("VERSION")
                .help("Python version to use. If not specified, it will be automatically select the latest supported version"),
        )
        .arg(
            Arg::new("torch")
                .short('t')
                .long("torch")
                .value_name("VERSION")
                .help("PyTorch version to use. If not specified, it will be automatically select the latest supported version"),
        )
        .arg(
            Arg::new("cuda")
                .short('c')
                .long("cuda")
                .value_name("VERSION")
                .help("CUDA version to use. If not specified, it will be automatically select the latest supported version"),
        )
        .arg(
            Arg::new("region")
                .short('r')
                .long("region")
                .value_name("REGION")
                .help("Ubuntu mirror region to use. If not specified, it will be automatically selected closest to you"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(ArgAction::SetTrue)
                .help("Verbose output"),
        )
        .arg(
            Arg::new("light")
                .short('l')
                .long("light")
                .action(ArgAction::SetTrue)
                .help("Use light version of the image"),
        );

    let matches = command.get_matches();

    let python_version = matches.get_one::<String>("python").map(|v| {
        let python_parts = v.split('.').collect::<Vec<_>>();
        format!("cp{}{}", python_parts[0], python_parts[1])
    });
    let torch_version = matches.get_one::<String>("torch").map(|v| v.to_string());
    let mut cuda_version = matches.get_one::<String>("cuda").map(|v| {
        let cuda_parts = v.split('.').collect::<Vec<_>>();
        format!("cu{}{}", cuda_parts[0], cuda_parts[1])
    });
    let _region = matches.get_one::<String>("region").map(|v| v.to_string());
    let verbose = matches.get_flag("verbose");
    let _light = matches.get_flag("light");

    if cuda_version.is_none() && env::consts::OS == "macos" {
        cuda_version = Some("cpu".to_string());
    }

    let resolver = VersionResolver::new(verbose);
    let versions = resolver.resolve_versions(python_version, torch_version, cuda_version).await;
    let version = versions.last().unwrap();
    let base_image = resolver.resolve_image_tag(version).await;

    let client = DockerClient::new(verbose);

    println!(
        "[+] Using resolved versions: python {}, torch {}, cuda {}, and tag {}",
        version.get_python_semantic_version(),
        version.torch,
        version.get_accelerator_semantic_version(),
        base_image
    );

    let containers = client.list_containers().await;

    for container in containers {
        if let Some(image) = container.image {
            if image == base_image {
                let container_id = container.id.unwrap();
                client.remove_container(container_id.clone().as_str()).await;
            }
        }
    }

    client.run_container("cudalis_setup", base_image.as_str()).await;
    client.execute_commands(
        "cudalis_setup",
        vec![
            "echo 'export DEBIAN_FRONTEND=noninteractive' >> /etc/bash.bashrc",
            "echo 'deb https://ftp.udx.icscoe.jp/Linux/ubuntu bionic main universe' >> /etc/apt/sources.list",
            "echo 'deb https://ftp.udx.icscoe.jp/Linux/ubuntu bionic-security main universe' >> /etc/apt/sources.list",
            "echo 'deb https://ftp.udx.icscoe.jp/Linux/ubuntu bionic-updates main universe' >> /etc/apt/sources.list",
            "sed -i.bak -r 's@http://(jp\\.)?archive\\.ubuntu\\.com/ubuntu/?@https://ftp.udx.icscoe.jp/Linux/ubuntu/@g' /etc/apt/sources.list",
            "apt update -y",
            "apt upgrade -y",
            "apt install -y --allow-downgrades git python3-pip software-properties-common packagekit policykit-1 libpam-systemd systemd systemd-sysv libsystemd0=245.4-4ubuntu3.21 networkd-dispatcher",
            "add-apt-repository -y ppa:deadsnakes/ppa",
            "apt update -y",
            &format!(
                "apt install -y python{} python{}-dev python-{}-venv",
                version.get_python_semantic_version(),
                version.get_python_semantic_version(),
                version.get_python_semantic_version(),
            ),
            "apt autoremove -y",
            "apt clean",
            &format!(
                "update-alternatives --install /usr/bin/python python /usr/bin/python{} 1",
                version.get_python_semantic_version(),
            ),
            &format!(
                "pip install torch=={} -f https://download.pytorch.org/whl/{}",
                version.torch,
                version.accelerator,
            ),
            &format!(
                "pip install torchvision torchaudio -f https://download.pytorch.org/whl/{}",
                version.accelerator,
            ),
            "mkdir /app",
        ],
    ).await;

    let image_name = format!(
        "cudalis:{}-{}-{}",
        version.get_python_semantic_version(),
        version.torch,
        version.get_accelerator_semantic_version()
    );

    client.commit_container("cudalis_setup", &image_name).await;
    client.remove_container("cudalis_setup").await;
    client.remove_image(base_image.as_str()).await;

    println!("[+] Done. You can now use the image with tag: {}", image_name);
}
