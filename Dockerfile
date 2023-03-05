# Use the official Ubuntu 18.04 as base
FROM continuumio/miniconda3:latest

RUN apt update && apt install -y gcc && apt install -y g++ && apt install nano && apt install -y git && git clone https://github.com/mitchelldehaven/bevers && cd bevers && apt install libsqlite3-dev && bash setup.sh
WORKDIR bevers 
COPY fever/ fever/
COPY pubmed/ pubmed/
COPY scifact/ scifact/
COPY models/ models/
