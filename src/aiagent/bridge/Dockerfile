FROM	python:3.12-alpine

WORKDIR	/app
COPY	requirements.txt .
RUN	pip install --no-cache-dir -r requirements.txt
COPY	bridge.py retriever.py ./

EXPOSE	9092
CMD	["python", "bridge.py"]
