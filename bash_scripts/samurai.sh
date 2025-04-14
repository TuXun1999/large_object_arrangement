## Install samurai (for automatic object image collection purpose)
## (Optional)
git clone https://github.com/yangchris11/samurai.git
cd samurai
cd sam2
pip install -e .
pip install loguru
cd checkpoints && \
./download_ckpts.sh && \
cd ..
