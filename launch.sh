pushd pdr-ms
uvicorn main:app --port 8000 --host localhost &
popd

pushd frontend
PDR_MS_HOSTNAME=localhost PDR_MS_PORT=8000 streamlit run main.py &
popd