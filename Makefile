#!make
.PHONY: docker-build docker-run docker

IMAGE_NAME = chest-segment
IMAGE_TAG = 0.1

USER_NAME = $(shell whoami)
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)


docker-build:
	docker build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.


docker-run:
	docker run -it \
		--shm-size=64Gb \
		--net host \
		--gpus all \
		-v $(PWD):/home/${USER_NAME}/chest_segment \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


docker: docker-build docker-run
