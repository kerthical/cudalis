use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, LogOutput, RemoveContainerOptions, StartContainerOptions};
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::image::{CommitContainerOptions, CreateImageOptions};
use bollard::models::ContainerSummary;
use bollard::Docker;
use futures_util::StreamExt;
use std::process::exit;
use std::time::Duration;

pub struct DockerClient {
    docker: Docker,
    verbose: bool,
}

impl DockerClient {
    pub(crate) fn new(verbose: bool) -> Self {
        let mut docker = Docker::connect_with_local_defaults().unwrap();
        docker.set_timeout(Duration::from_secs(3600));
        DockerClient { docker, verbose }
    }

    pub(crate) async fn list_containers(&self) -> Vec<ContainerSummary> {
        self.docker
            .list_containers(Some(ListContainersOptions::<String> {
                all: true,
                ..Default::default()
            }))
            .await
            .unwrap()
    }

    pub(crate) async fn run_container(&self, container_name: &str, image_name: &str) {
        println!("[+] Running container {} with image {}", container_name, image_name);

        let stream = self.docker.create_image(
            Some(CreateImageOptions {
                from_image: image_name,
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

        self.docker
            .create_container(
                Some(CreateContainerOptions {
                    name: container_name,
                    platform: None,
                }),
                Config {
                    image: Some(image_name),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    tty: Some(true),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        self.docker
            .start_container(container_name, None::<StartContainerOptions<String>>)
            .await
            .unwrap();
    }

    pub(crate) async fn remove_container(&self, container_name: &str) {
        println!("[+] Removing container {}", container_name);
        self.docker
            .remove_container(
                container_name,
                Some(RemoveContainerOptions {
                    force: true,
                    v: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }

    pub(crate) async fn commit_container(&self, container_name: &str, image_name: &str) {
        println!("[+] Committing container {} to image {}", container_name, image_name);
        self.docker
            .commit_container(
                CommitContainerOptions {
                    container: container_name,
                    repo: image_name,
                    ..Default::default()
                },
                Config::<String>::default(),
            )
            .await
            .unwrap();
    }

    pub(crate) async fn remove_image(&self, image_name: &str) {
        println!("[+] Removing image {}", image_name);
        self.docker.remove_image(image_name, None, None).await.unwrap();
    }

    pub(crate) async fn execute_commands(&self, container_id: &str, commands: Vec<&str>) {
        for command in commands {
            self.execute_command(container_id, command).await;
        }
    }

    pub(crate) async fn execute_command(&self, container_id: &str, command: &str) {
        if self.verbose {
            println!("[+] {}", command);
        } else {
            print!("[+] {}", command);
        }

        let id = self
            .docker
            .create_exec(
                container_id,
                CreateExecOptions {
                    cmd: Some(vec!["bash", "-c", command]),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    tty: Some(true),
                    env: Some(vec!["DEBIAN_FRONTEND=noninteractive"]),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .id;

        let stream = self
            .docker
            .start_exec(
                &id,
                Some(StartExecOptions {
                    detach: false,
                    ..Default::default()
                }),
            )
            .await;

        let mut printed = false;
        if let Ok(StartExecResults::Attached { mut output, input: _ }) = stream {
            while let Some(result) = output.next().await {
                match result {
                    Ok(LogOutput::StdOut { message }) => {
                        if self.verbose {
                            print!("{}", String::from_utf8_lossy(&message));
                            printed = true;
                        }
                    }
                    Ok(LogOutput::StdErr { message }) => {
                        eprint!("{}", String::from_utf8_lossy(&message));
                        printed = true;
                    }
                    _ => {}
                }
            }
        }

        if self.verbose {
            if printed {
                println!();
            }
        } else if printed {
            println!(" done");
        } else {
            println!();
        }
    }
}
