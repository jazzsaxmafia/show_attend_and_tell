# Show, attend and tell에 데이터 구축 코드 포함시키고, 한글 주석 달아놓은 것.
# arctic-captions

* vgg net의 14*14*512 -> 14*14는 feature window 크기, 512개 feature map
function: init_params()

Wemb : [vocab size, dim_word]. vocab size는 사전 내 단어 수. dim_word는 word embedding의 차원

function: param_init_lstm()
ctx_dim은 이미지 윈도우 개수. 여기서는 14*14*512중 512차원. dim은 hidden size
encoder는 왼쪽 -> 오른쪽으로 읽어나가고 encoder_rev는 오른쪽->왼쪽으로 읽어나감

### optional ###
* 이미지를 hidden으로 embedding
encoder_W: [ctx_dim, dim]
encoder_U: [dim, dim]
encoder_b: [dim]

encoder_rev_W: [ctx_dim, dim]
encoder_rev_U: [dim, dim]
encoder_rev_b: [dim]

이걸 했다면 ctx는 encoding된 상태이므로 ctx_dim은 dim*2 (encoder+encoder_rev)
##############
n_in = dim_word, dim=dim, dimctx=ctx_dim

word -> hidden으로 embedding
decoder_W: [dim_word, dim * 4] (ifog)
decoder_U : [dim, dim * 4]
decoder_b : [dim * 4] 

image -> hidden으로 embedding
decoder_Wc : [ctx_dim,  dim*4]
decoder_Wc_att : [ctx_dim, ctx_dim]
decoder_Wd_att : [dim, ctx_dim]
decoder_b_att
decoder_U_att : [dim_ctx, 1]
decoder_c_att : [1]

### optional ###
decoder_W_sel
decoder_b_sel
#############

ff_state_W: [ctx_dim, dim]
ff_state_b: [dim]

ff_memory_W: [ctx_dim, dim]
ff_memory_b : [dim]

hidden -> word 생성부분
ff_logit_lstm_W : [dim, dim_word]
ff_logit_lstm_b [dim_word]


Source code for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)
runnable on GPU and CPU.

Joint collaboration between the Université de Montréal & University of Toronto.

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* A relatively recent version of [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [argparse](https://www.google.ca/search?q=argparse&oq=argparse&aqs=chrome..69i57.1260j0j1&sourceid=chrome&es_sm=122&ie=UTF-8#q=argparse+pip)

In addition, this code is built using the powerful
[Theano](http://www.deeplearning.net/software/theano/) library. If you
encounter problems specific to Theano, please use a commit from around
February 2015 and notify the authors.

To use the evaluation script (metrics.py): see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Reference

If you use this code as part of any published research, please acknowledge the
following paper (it encourages researchers who publish their code!):

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *To appear ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    } 

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).
