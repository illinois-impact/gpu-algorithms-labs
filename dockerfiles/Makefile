

ARCH := $(shell uname -m)

ifeq (${ARCH}, x86_64)
  ARCH = amd64
else ifeq(${ARCH}, "ppc64le")
  ARCH = ppc64le
else
  exit 1
endif

all: ${ARCH}

docker_login:
	@echo "You will need to log in with a docker hub account that has write permission to raiproject"
	docker login

amd64: amd64_cuda91 amd64_cuda100

amd64_cuda91: docker_login Dockerfile.amd64
	docker build -f Dockerfile.amd64 -t raiproject/pumps2018:amd64-cuda91 .
	docker push raiproject/pumps2018:amd64-cuda91

amd64_cuda100: docker_login Dockerfile.amd64_cuda100
	docker build -f Dockerfile.amd64_cuda100 -t raiproject/pumps2018:amd64-cuda100 .
	docker push raiproject/pumps2018:amd64-cuda100

ppc64le: docker_login Dockerfile.ppc64le
	docker build -f Dockerfile.ppc64le -t raiproject/pumps2018:ppc64le-cuda91 .
	docker push raiproject/pumps2018:ppc64le-cuda91