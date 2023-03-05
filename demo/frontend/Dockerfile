FROM node:latest as node
RUN apt update && apt install nano && apt install -y git && git clone https://github.com/mitchelldehaven/bevers && cd bevers/demo/frontend 
WORKDIR bevers/demo/frontend
RUN npm install
RUN npm install -g @angular/cli
CMD ng serve --host 0.0.0.0 --port 4200