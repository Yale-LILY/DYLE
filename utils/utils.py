import torch
import os
import shutil
from pyrouge import Rouge155
from tqdm import tqdm
import logging
import subprocess as sp
from config import Config
logging.basicConfig(level=logging.INFO)

config = Config()


def gpu_wrapper(item, device=None):
    if config.gpu:
        # print(item)
        if device is not None:
            device = torch.device("cuda:{}".format(device))
            return item.to(device)
        else:
            return item.cuda()
    else:
        return item


def pretty_string(flt):
    ret = '%.6f' % flt
    if flt >= 0:
        ret = "+" + ret
    return ret


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def rouge_with_pyrouge(preds, refs):
    assert len(preds) == len(refs)

    # Dump refs.
    tmp_ref_dir = os.path.join(config.tmp_dir, 'refs')
    if os.path.exists(tmp_ref_dir):
        shutil.rmtree(tmp_ref_dir)
    os.mkdir(tmp_ref_dir)
    for i, ref in enumerate(refs):
        with open(os.path.join(tmp_ref_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe(ref.strip()))  # sentences in ref are joined with \n

    # Dump preds.
    tmp_pred_dir = os.path.join(config.tmp_dir, 'preds')
    if os.path.exists(tmp_pred_dir):
        shutil.rmtree(tmp_pred_dir)
    os.mkdir(tmp_pred_dir)
    for i, pred in enumerate(preds):
        with open(os.path.join(tmp_pred_dir, '{}.dec'.format(i)), 'w') as f:
            f.write(make_html_safe(pred.strip()))  # sentences in pred are joined with \n

    dec_pattern = r'(\d+).dec'
    ref_pattern = '#ID#.ref'
    cmd = '-c 95 -r 1000 -n 2 -m'
    system_id = 1
    _ROUGE_PATH = 'utils/ROUGE-1.5.5'
    logging.disable(level=logging.INFO)  # Disable the huge amount of file-processing logging.

    # Calculate rouge.
    tmp_rouge_dir = os.path.join(config.tmp_dir, 'rouge')
    if os.path.exists(tmp_rouge_dir):
        shutil.rmtree(tmp_rouge_dir)
    os.mkdir(tmp_rouge_dir)
    Rouge155.convert_summaries_to_rouge_format(
        tmp_pred_dir, os.path.join(tmp_rouge_dir, 'dec'))
    Rouge155.convert_summaries_to_rouge_format(
        tmp_ref_dir, os.path.join(tmp_rouge_dir, 'ref'))

    # Note that model summaries are the reference summaries,
    # which seems weird but see the following lines extracted from the README of ROUGE-1.5.5:
    # ''The first file path is the peer summary (system summary) and it
    # follows with a list of model summaries (reference summaries) separated
    # by white spaces (spaces or tabs).''

    Rouge155.write_config_static(
        system_dir=os.path.join(tmp_rouge_dir, 'dec'), system_filename_pattern=dec_pattern,
        model_dir=os.path.join(tmp_rouge_dir, 'ref'), model_filename_pattern=ref_pattern,
        config_file_path=os.path.join(tmp_rouge_dir, 'settings.xml'), system_id=system_id
    )
    cmd = (os.path.join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
           + ' -e {} '.format(os.path.join(_ROUGE_PATH, 'data'))
           + cmd
           + ' -a {}'.format(os.path.join(tmp_rouge_dir, 'settings.xml')))
    output = sp.check_output(cmd.split(' '), universal_newlines=True)

    print('\n\n\n')
    print(output)

    logging.disable(level=logging.INFO)  # Re-enable logging.

    # Remove temp files.
    shutil.rmtree(tmp_ref_dir)
    shutil.rmtree(tmp_pred_dir)
    shutil.rmtree(tmp_rouge_dir)

    index1 = output.index('ROUGE-1 Average_F: ') + len('ROUGE-1 Average_F: ')
    rouge1 = float(output[index1: index1 + 7])
    index2 = output.index('ROUGE-2 Average_F: ') + len('ROUGE-2 Average_F: ')
    rouge2 = float(output[index2: index2 + 7])
    indexL = output.index('ROUGE-L Average_F: ') + len('ROUGE-L Average_F: ')
    rougeL = float(output[indexL: indexL + 7])

    return rouge1, rouge2, rougeL
