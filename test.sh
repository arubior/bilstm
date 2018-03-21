 pylint bilstm -f parseable -r n --extension-pkg-whitelist=cv2
 pycodestyle bilstm --max-line-length=120
 pydocstyle bilstm
 pytest --cov=bilstm --cov-report xml tests
 coverage report -m
