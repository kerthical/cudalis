use std::env;
use std::process::exit;

pub struct Version {
    pub name: String,
    pub torch: String,
    pub python: String,
    pub accelerator: String,
    pub os: String,
}

impl Version {
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
        let torch = package_parts.get(1)?.split("%2B").next()?.to_string();
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

        Some(Version {
            name,
            torch,
            python,
            os,
            accelerator,
        })
    }

    pub(crate) fn get_python_semantic_version(&self) -> String {
        let python_version = self.python.replace("cp", "");
        let major = python_version.chars().nth(0).unwrap();
        let minor = python_version.chars().nth(1).unwrap();
        format!("{}.{}", major, minor)
    }

    pub(crate) fn get_accelerator_semantic_version(&self) -> String {
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

pub struct VersionResolver {
    verbose: bool,
}

impl VersionResolver {
    pub(crate) fn new(verbose: bool) -> Self {
        VersionResolver { verbose }
    }

    pub(crate) async fn resolve_versions(
        &self,
        python_version: Option<String>,
        torch_version: Option<String>,
        cuda_version: Option<String>,
    ) -> Vec<Version> {
        println!(
            "[+] Resolving versions with python {}, torch {}, and cuda {}",
            python_version.as_ref().unwrap_or(&"latest".to_string()),
            torch_version.as_ref().unwrap_or(&"latest".to_string()),
            cuda_version.as_ref().unwrap_or(&"latest".to_string())
        );

        let result = reqwest::get("https://download.pytorch.org/whl/torch_stable.html")
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        let mut versions = result
            .split('\n')
            .filter_map(Version::parse_from_html_tag)
            .filter(|v| v.name == "torch")
            .collect::<Vec<_>>();

        if self.verbose {
            println!("    Found {} versions", versions.len());
        }

        versions = self.filter_versions_by_os_and_arch(versions);

        if self.verbose {
            println!("    Found {} versions after filtering by OS and architecture", versions.len());
        }

        versions = self.filter_versions_by_specified_version(versions, python_version, |v| &v.python);

        if self.verbose {
            println!("    Found {} versions after filtering by Python version", versions.len());
        }

        versions = self.filter_versions_by_specified_version(versions, torch_version, |v| &v.torch);

        if self.verbose {
            println!("    Found {} versions after filtering by Torch version", versions.len());
        }

        versions = self.filter_versions_by_specified_version(versions, cuda_version, |v| &v.accelerator);

        if self.verbose {
            println!("    Found {} versions after filtering by CUDA version", versions.len());
            println!();
        }

        versions.sort_by(|a, b| a.torch.cmp(&b.torch));

        if versions.is_empty() {
            eprintln!("[!] No versions found with the specified constraints");
            exit(1);
        }

        versions
    }

    pub(crate) async fn resolve_image_tag(&self, version: &Version) -> String {
        if version.accelerator == "cpu" {
            "ubuntu:22.04".to_string()
        } else {
            println!("[+] Resolving image tag with cuda {}", version.get_accelerator_semantic_version());
            let result: serde_json::Value = reqwest::get(&format!(
                "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100&name={}",
                version.get_accelerator_semantic_version()
            ))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

            if self.verbose {
                println!("    Found {} tags", result["count"].as_u64().unwrap());
            }

            let tags = result["results"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|t| {
                    let tag = t["name"].as_str().unwrap().to_string();

                    tag.starts_with(&version.get_accelerator_semantic_version()) && tag.contains("ubuntu") && tag.contains("devel")
                })
                .collect::<Vec<_>>();

            if self.verbose {
                println!("    Found {} tags after filtering by CUDA version", tags.len());
                println!();
            }

            let tag = tags.iter().max_by(|a, b| {
                let a_tag = a["name"].as_str().unwrap().to_string();
                let b_tag = b["name"].as_str().unwrap().to_string();

                a_tag.cmp(&b_tag)
            });

            if tag.is_none() {
                eprintln!("[!] No CUDA image found for version {}", version.get_accelerator_semantic_version());
                exit(1);
            }

            format!("nvidia/cuda:{}", tag.unwrap()["name"].as_str().unwrap())
        }
    }

    fn filter_versions_by_os_and_arch(&self, versions: Vec<Version>) -> Vec<Version> {
        let computer_os = env::consts::OS.to_lowercase();
        let computer_arch = env::consts::ARCH.to_lowercase();

        versions
            .into_iter()
            .filter(|v| v.os.contains(&computer_os) && v.os.contains(&computer_arch))
            .collect()
    }

    fn filter_versions_by_specified_version<F>(
        &self,
        versions: Vec<Version>,
        specified_version: Option<String>,
        version_extractor: F,
    ) -> Vec<Version>
    where
        F: Fn(&Version) -> &String,
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
}
