
def cal_aspect_prf(goldens, predicts, num_aspects, verbal=False):
    """

    :param verbal:
    :param num_aspects:
    :param list of models.AspectOutput goldens:
    :param list of models.AspectOutput predicts:
    :return:
    """
    tp = [0] * num_aspects
    fp = [0] * num_aspects
    fn = [0] * num_aspects

    for g, p in zip(goldens, predicts):
        for i in range(num_aspects):
            if g.scores[i] == p.scores[i] == 1:
                tp[i] += 1
            elif g.scores[i] == 1:
                fn[i] += 1
            elif p.scores[i] == 1:
                fp[i] += 1

    p = [tp[i]/(tp[i]+fp[i]) for i in range(num_aspects)]
    r = [tp[i]/(tp[i]+fn[i]) for i in range(num_aspects)]
    f1 = [2*p[i]*r[i]/(p[i]+r[i]) for i in range(num_aspects)]

    micro_p = sum(tp)/(sum(tp)+sum(fp))
    micro_r = sum(tp)/(sum(tp)+sum(fn))
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)

    macro_p = sum(p)/num_aspects
    macro_r = sum(r)/num_aspects
    macro_f1 = sum(f1)/num_aspects

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)
        print('micro:', (micro_p, micro_r, micro_f1))
        print('macro:', (macro_p, macro_r, macro_f1))

    return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)
