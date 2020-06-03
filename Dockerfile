# base image
FROM ubuntu:18.04

# update package list
RUN apt update && apt upgrade -y
# install python3.6 and pip3
RUN apt install python3.6-dev -y
RUN apt install python3-pip -y

# install git
RUN apt install git -y

# 로컬의 requirements.txt를 컨테이너의 home 폴더에 복사한다
COPY ./resources/requirements.txt /home/

# install packages
RUN pip3 install -r /home/requirements.txt
# /root/.jupyter/jupyter_notebook_config.py 파일 생성
RUN jupyter notebook --generate-config
# 브라우저을 열지않게 설정
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
# 비밀번호 설정 (접속할 때 실제 사용하는 비밀번호에서 암호화 된 비밀번호를 써야함)
RUN echo "c.NotebookApp.password = u'sha1:ec859a1f6ca9:1caf8db244fd4cdc7bf8f0c1f052d6b8438e9ab9'" >> /root/.jupyter/jupyter_notebook_config.py
# 주피터노트북 테마설정
RUN mkdir /root/.jupyter/custom
RUN echo '.container { width:100% !important; }\n' >> /root/.jupyter/custom/custom.css

# home에서 jupyter notebook 실행 (루트 권한을 주고 8888포트에서 실행한다)
WORKDIR /home/
ENTRYPOINT jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
