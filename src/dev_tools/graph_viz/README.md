# Graph GIF tools

```sh
cd /mnt/fileserver/prj/dmctp/user/venv
TENSOR_BACKEND=generic ./run ../src/tensor/src/dev_tools/graph_viz/test/lifecycle_demo.c

python3 ../src/tensor/src/dev_tools/graph_viz/vtensor_lifetime.py \
  --poll-ms 1 --frame-ms 900 --trace /tmp/vtensor-trace.csv \
  --output ../src/tensor/src/dev_tools/graph_viz/test/vtensor_lifecycle.gif \
  --launch -- ./bin/lifecycle_demo

python3 ../src/tensor/src/dev_tools/graph_viz/tensor_lifetime.py \
  --poll-ms 1 --frame-ms 900 --trace /tmp/tensor-trace.csv \
  --output ../src/tensor/src/dev_tools/graph_viz/test/tensor_lifecycle.gif \
  --launch -- ./bin/lifecycle_demo
```

Requires Python 3, Pillow, and Graphviz `dot`.
