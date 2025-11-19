# run in the same folder:
# sudo docker image build ./docker --tag meetingdocker/ml:refinedgae
FROM meetingdocker/ml:pt_0.3

# RUN conda install geoopt==0.5.0 \
# matplotlib==2.2.3 \
# networkx==2.5 \
# numpy==1.19.0 \
# scikit_learn==1.3.2 \
# scipy==1.4.1

# RUN pip install geoopt scikit_learn

# RUN conda install torch_geometric==2.3.1 
# RUN conda install pandas
RUN conda install -c dglteam/label/th22_cu121 dgl

# Fix NumPy compatibility issue - downgrade to 1.x
RUN pip install "numpy<2" --force-reinstall

# Install pandas (required by DGL)
RUN pip install pandas

# Install compatible version of torchdata with datapipes support
RUN pip install torchdata==0.7.1 portalocker

# Install Open Graph Benchmark
RUN pip install ogb


