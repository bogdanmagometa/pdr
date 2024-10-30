export CHILD_DOCS_DIR=$(realpath chroma_persist)
export PARENT_DOCS_DIR=$(realpath store)
pushd pdr-ms
uvicorn main:app --port 8000 --host localhost &
popd

pushd frontend
PDR_MS_HOSTNAME=localhost PDR_MS_PORT=8000 streamlit run --server.address localhost --server.port 8080 main.py &
popd
