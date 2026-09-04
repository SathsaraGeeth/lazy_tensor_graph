# JIT cache GIF

Run the same tensor program twice. Run 1 fills the RAM cache and stores the
persistent cache; run 2 loads that file and shows cache hits.

```sh
cd /mnt/fileserver/prj/dmctp/user/venv
python3 ../src/tensor/src/dev_tools/graph_viz/jit_lifetime.py \
  --frame-ms 900 --output ../src/tensor/src/dev_tools/graph_viz/test/jit_lifecycle.gif \
  --launch -- ./bin/lifecycle_demo
```

The GIF shows lookup, miss, compilation, hit, RAM-cache reuse, and persistent
cache store/load. Requires Python 3, Pillow, and Graphviz `dot`.
