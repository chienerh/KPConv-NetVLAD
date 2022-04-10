build image
```
docker build --tag umcurly/nvidia_pytorch_21_12 .
```
build container
```
bash build_docker_container.bash [container_name]
```
run docker
```
docker exec -it [container_name] /bin/bash
```

run docker with root access
```
docker exec -u root -it [container_name] /bin/bash
```

