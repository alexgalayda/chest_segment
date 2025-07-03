#!make
.PHONY: docker-build-cpu docker-run-cpu docker-build-gpu docker-run-gpu docker-gpu podman-build-gpu podman-run-gpu podman-gpu

IMAGE_NAME = chest-segment
# IMAGE_TAG = 0.2-gpu
IMAGE_TAG = 0.2-gpu-podman

USER_NAME = $(shell whoami)
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)


docker-build-gpu:
	docker build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg BUILD_TARGET=gpu \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.


docker-run-gpu:
	docker run -it \
		--shm-size=64Gb \
		--net host \
		--gpus all \
		-v $(PWD):/home/${USER_NAME}/chest_segment \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


docker-gpu: docker-build-gpu docker-run-gpu


docker-build-cpu:
	docker build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg BUILD_TARGET=cpu \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.

docker-run-cpu:
	docker run -it \
		--shm-size=64Gb \
		--net host \
		-v $(PWD):/home/${USER_NAME}/chest_segment \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


docker-cpu: docker-build-cpu docker-run-cpu


podman-build-gpu:
	podman build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg BUILD_TARGET=gpu \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.


podman-run-gpu:
	podman run -it \
		--userns=keep-id \
		--shm-size=64g \
		--net host \
		--hooks-dir=/usr/share/containers/oci/hooks.d \
		--security-opt=label=disable \
		--device nvidia.com/gpu=all \
		-v $(PWD):/home/${USER_NAME}/chest_segment \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


podman-gpu: podman-build-gpu podman-run-gpu
