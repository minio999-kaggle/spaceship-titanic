pipeline {
    agent {
        kubernetes {
            label 'jenkins-docker-pod'
            defaultContainer 'docker'
            yaml """
apiVersion: v1
kind: Pod
metadata:
labels:
  component: ci
spec:
  # Use service account that can deploy to all namespaces
  serviceAccountName: jenkins
  containers:
  - name: docker
    image: docker:latest
    command:
    - cat
    tty: true
    volumeMounts:
    - mountPath: /var/run/docker.sock
      name: docker-sock
  volumes:
    - name: docker-sock
      hostPath:
        path: /var/run/docker.sock
"""
        }
    }

    triggers {
        pollSCM('*/10 * * * *')
    }

    environment {
        def container = null
        def dockerfile = "."
        def dockerRegistryURL = "127.0.0.1:5000"
        def imageName = "demoapp-${BRANCH_NAME}"
        def masterBranch = "master"
        def qaBranch = "qa"
    }

    stages {
        stage('building') {
            steps {
                script {
                    docker.withRegistry(dockerRegistryURL) {
                        container = docker.build(imageName, dockerfile)
                    }
                }
            }
        }

        stage('linting') {
            steps {
                script {
                    container.inside("-u root --entrypoint=\'\'") {
                        sh 'chmod +x /app/linter.sh'
                        sh '/app/linter.sh "9.00" "/src/app"'
                    }
                }
            }
        }

        stage('testing') {
            when {
                anyOf {
                    expression { env.BRANCH_NAME == devBranch }
                }
            }
            steps {
                script {
                    container.inside("-u root --entrypoint=\'\'") {
                        sh 'cd /app && pytest -n 2'
                    }
                }
            }
        }
    }
    post {
        always {
            cleanWs()

            script {
                if ( env.BRANCH_NAME == masterBranch || env.BRANCH_NAME == qaBranch ) {
                    sh "docker image rm $imageName"
                }
                sh "docker container prune -f"
                sh "docker image prune -f"
                sh "docker network prune -f"
                sh "docker volume prune -f"
            }
        }
    }
}