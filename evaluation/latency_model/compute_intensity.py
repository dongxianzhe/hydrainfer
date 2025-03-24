def vision_model(
    n,  # num images
    c,  # channels
    h,  # vision model hidden_size
    H,  # language model hidden_size
    t,  # num of image tokens per image
    i,  # intermedia size
    d,  # head dimension
    q,  # number of query head
    L,  # number of layers
    ):
    num_conv2d    = 1
    num_concat    = 1
    num_posembed  = 1
    num_layernorm = 2*L+2
    num_ffn       = L
    num_attn_proj = L
    num_attn      = L
    num_residual  = 2 * L
    num_projector = 1

    read_conv2d   =14*14*c*h+336*336*c*n
    read_concat   =576*n*h+h
    read_posembed =n*t*h
    read_layernorm=n*t*h+2*h
    read_ffn      =n*t*h+h*i+i+n*t*i+n*t*i+i*h+h
    read_attn_proj=4*h*h+4*h+n*t*h
    read_attn     =n*t*h+n*t*h
    read_residual =2*n*t*h
    read_projector=n*576*h+h*H+h+n*576*H+H*h+h

    write_conv2d   =n*t*h
    write_concat   =n*t*h
    write_posembed =n*t*h
    write_layernorm=n*t*h
    write_ffn      =3*n*t*h
    write_attn_proj=4*n*t*h
    write_attn     =n*t*h
    write_residual =n*t*h
    write_projector=n*576*h+n*576*H

    comp_conv2d    = num_conv2d    * ((2*14*14*c-1)*h*576*n)
    comp_concat    = num_concat    * (0)
    comp_posembed  = num_posembed  * (n*t*h)
    comp_layernorm = num_layernorm * (n*t*(6*h))
    comp_ffn       = num_ffn       * (n*t*i*2*h+n*t*i*6+n*t*h*2*i)
    comp_attn_proj = num_attn_proj * (4*n*t*h*2*h)
    comp_attn      = num_attn      * (n*q*t*t*(d-1)*2+n*q*t*(3*t-1))
    comp_residual  = num_residual  * (n*t*h)
    comp_projector = num_projector * (n*576*H*2*h+n*576*h*2*H)

    mem_conv2d    = num_conv2d    * (read_conv2d    + write_conv2d)
    mem_concat    = num_concat    * (read_concat    + write_concat )
    mem_posembed  = num_posembed  * (read_posembed  + write_posembed)
    mem_layernorm = num_layernorm * (read_layernorm + write_layernorm)
    mem_ffn       = num_ffn       * (read_ffn       + write_ffn)
    mem_attn_proj = num_attn_proj * (read_attn_proj + write_attn_proj)
    mem_attn      = num_attn      * (read_attn      + write_attn)
    mem_residual  = num_residual  * (read_residual  + write_residual)
    mem_projector = num_projector * (read_projector + write_projector)

    return sum([
        comp_conv2d, 
        comp_concat, 
        comp_posembed, 
        comp_layernorm, 
        comp_ffn, 
        comp_attn_proj, 
        comp_attn, 
        comp_residual, 
        comp_projector
    ]), sum([
        mem_conv2d, 
        mem_concat, 
        mem_posembed, 
        mem_layernorm, 
        mem_ffn, 
        mem_attn_proj, 
        mem_attn, 
        mem_residual, 
        mem_projector, 
    ]) * 2


def language_model(
    h,  # hidden size
    d,  # head dim
    q,  # num q heads
    k,  # num k heads
    V,  # vocabulary size
    i,  # intermediate size
    n,  # n tokens
    s,  # n kv cache tokens
    L,  # n layers
):
    num_embed        = 1
    num_rmsnorm      = L * 2 + 1
    num_residual     = L * 2
    num_rope         = L
    num_set_kv_cache = L
    num_head         = 1
    num_ffn          = L
    num_attnproj     = L
    num_attn         = L

    read_embed       =n+n*h
    read_rmsnorm     =h+n*h
    read_residual    =2*n*h
    read_rope        =3*n*h+n
    read_set_kv_cache=2*n*h+n
    read_head        =n*h+h*V
    read_ffn         =n*h+h*i+n*h+h*i+n*i+2*n*i+n*i+i*h
    read_attnproj    =4*n*h+4*h*h
    read_attn        =n*h+s*h

    write_embed       =n*h
    write_rmsnorm     =n*h
    write_residual    =n*h
    write_rope        =2*n*h
    write_set_kv_cache=2*n*h
    write_head        =n*V
    write_ffn         =n*i+n*i+n*i+n*i+n*h
    write_attnproj    =4*n*h
    write_attn        =n*h

    mem_embed        = num_embed        * (read_embed       + write_embed)
    mem_rmsnorm      = num_rmsnorm      * (read_rmsnorm     + write_rmsnorm)
    mem_residual     = num_residual     * (read_residual    + write_residual)
    mem_rope         = num_rope         * (read_rope        + write_rope)
    mem_set_kv_cache = num_set_kv_cache * (read_set_kv_cache+ write_set_kv_cache)
    mem_head         = num_head         * (read_head        + write_head)
    mem_ffn          = num_ffn          * (read_ffn         + write_ffn)
    mem_attnproj     = num_attnproj     * (read_attnproj    + write_attnproj)
    mem_attn         = num_attn         * (read_attn        + write_attn)

    comp_embed        = num_embed        * (0)
    comp_rmsnorm      = num_rmsnorm      * (n*4*h+n)
    comp_residual     = num_residual     * (n*h)
    comp_rope         = num_rope         * (6*n*h)
    comp_set_kv_cache = num_set_kv_cache * (0)
    comp_head         = num_head         * (n*V*(2*h-1))
    comp_ffn          = num_ffn          * (n*i*(2*h-1)*2+5*n*i+n*h*(2*i-1))
    comp_attnproj     = num_attnproj     * (4*n*h*(2*h-1))
    comp_attn         = num_attn         * (q*n*s*(2*d-1)+q*n*(3*s-1)+n*h*(2*s-1))

    return sum([
        comp_embed, 
        comp_rmsnorm, 
        comp_residual, 
        comp_rope, 
        comp_set_kv_cache, 
        comp_head, 
        comp_ffn, 
        comp_attnproj, 
        comp_attn]
    ), sum([
        mem_embed, 
        mem_rmsnorm, 
        mem_residual, 
        mem_rope, 
        mem_set_kv_cache, 
        mem_head, 
        mem_ffn, 
        mem_attnproj, 
        mem_attn]
    ) * 2

if __name__ == '__main__':
    for j in range(4096):
        l_comp, l_mem = language_model(
            h = 4096 , 
            d = 128  , 
            q = 32   , 
            k = 32   , 
            V = 32064, 
            i = 11008, 
            n = j , 
            s = 1024 , 
            L = 32, 
        )
        for i in range(17):
            v_comp, v_mem = vision_model(
                n = i, 
                c = 3, 
                h = 1024, 
                H = 4096, 
                t = 577, 
                i = 4096, 
                d = 64, 
                q = 16, 
                L = 24, 
            )
            intensity = (v_comp + l_comp) / (v_mem + l_mem)
            print(f'{intensity}', end=' ')
        print()