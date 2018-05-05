import torch
from torch.autograd import Variable
import codecs
import numpy as np
import math


PAD_token = 0
SOS_token = 1
EOS_token = 2
start_tags_with_attributes = ["<scr'+'ipt", "<!--", "<!DOCTYPE", "<a", "<abbr", "<acronym", "<address", "<applet", "<area", "<article", "<aside", "<audio", "<b", "<base", "<basefont", "<bdi", "<bdo", "<big", "<blockquote", "<body", "<br", "<button", "<canvas", "<caption", "<center", "<cite", "<code", "<col", "<colgroup", "<datalist", "<dd", "<del", "<details", "<dfn", "<dialog", "<dir", "<div", "<dl", "<dt", "<em", "<embed", "<fieldset", "<figcaption", "<figure", "<font", "<footer", "<form", "<frame", "<frameset", "<h1", "<head", "<header", "<hr", "<html", "<i", "<iframe", "<img", "<input", "<ins", "<kbd", "<label", "<legend", "<li", "<link", "<main", "<map", "<mark", "<menu", "<menuitem", "<meta", "<meter", "<nav", "<noframes", "<noscript", "<object", "<ol", "<optgroup", "<option", "<output", "<p", "<param", "<picture", "<pre", "<progress", "<q", "<rp", "<rt", "<ruby", "<s", "<samp", "<script", "<section", "<select", "<small", "<source", "<span", "<strike", "<strong", "<style", "<sub", "<summary", "<sup", "<table", "<tbody", "<td", "<template", "<textarea", "<tfoot", "<th", "<thead", "<time", "<title", "<tr", "<track", "<tt", "<u", "<ul", "<var",
                              "<video", "<wbr", "<SCR'+'IPT", "<!--", "<!DOCTYPE", "<A", "<ABBR", "<ACRONYM", "<ADDRESS", "<APPLET", "<AREA", "<ARTICLE", "<ASIDE", "<AUDIO", "<B", "<BASE", "<BASEFONT", "<BDI", "<BDO", "<BIG", "<BLOCKQUOTE", "<BODY", "<BR", "<BUTTON", "<CANVAS", "<CAPTION", "<CENTER", "<CITE", "<CODE", "<COL", "<COLGROUP", "<DATALIST", "<DD", "<DEL", "<DETAILS", "<DFN", "<DIALOG", "<DIR", "<DIV", "<DL", "<DT", "<EM", "<EMBED", "<FIELDSET", "<FIGCAPTION", "<FIGURE", "<FONT", "<FOOTER", "<FORM", "<FRAME", "<FRAMESET", "<H1", "<HEAD", "<HEADER", "<HR", "<HTML", "<I", "<IFRAME", "<IMG", "<INPUT", "<INS", "<KBD", "<LABEL", "<LEGEND", "<LI", "<LINK", "<MAIN", "<MAP", "<MARK", "<MENU", "<MENUITEM", "<META", "<METER", "<NAV", "<NOFRAMES", "<NOSCRIPT", "<OBJECT", "<OL", "<OPTGROUP", "<OPTION", "<OUTPUT", "<P", "<PARAM", "<PICTURE", "<PRE", "<PROGRESS", "<Q", "<RP", "<RT", "<RUBY", "<S", "<SAMP", "<SCRIPT", "<SECTION", "<SELECT", "<SMALL", "<SOURCE", "<SPAN", "<STRIKE", "<STRONG", "<STYLE", "<SUB", "<SUMMARY", "<SUP", "<TABLE", "<TBODY", "<TD", "<TEMPLATE", "<TEXTAREA", "<TFOOT", "<TH", "<THEAD", "<TIME", "<TITLE", "<TR", "<TRACK", "<TT", "<U", "<UL", "<VAR", "<VIDEO", "<WBR"]
end_tags = ["</scr'+'ipt>", "</!DOCTYPE>", "</a>", "</abbr>", "</acronym>", "</address>", "</applet>", "</area>", "</article>", "</aside>", "</audio>", "</b>", "</base>", "</basefont>", "</bdi>", "</bdo>", "</big>", "</blockquote>", "</body>", "</br>", "</button>", "</canvas>", "</caption>", "</center>", "</cite>", "</code>", "</col>", "</colgroup>", "</datalist>", "</dd>", "</del>", "</details>", "</dfn>", "</dialog>", "</dir>", "</div>", "</dl>", "</dt>", "</em>", "</embed>", "</fieldset>", "</figcaption>", "</figure>", "</font>", "</footer>", "</form>", "</frame>", "</frameset>", "</h1>", "</head>", "</header>", "</hr>", "</html>", "</i>", "</iframe>", "</img>", "</input>", "</ins>", "</kbd>", "</label>", "</legend>", "</li>", "</link>", "</main>", "</map>", "</mark>", "</menu>", "</menuitem>", "</meta>", "</meter>", "</nav>", "</noframes>", "</noscript>", "</object>", "</ol>", "</optgroup>", "</option>", "</output>", "</p>", "</param>", "</picture>", "</pre>", "</progress>", "</q>", "</rp>", "</rt>", "</ruby>", "</s>", "</samp>", "</script>", "</section>", "</select>", "</small>", "</source>", "</span>", "</strike>", "</strong>", "</style>", "</sub>", "</summary>", "</sup>", "</table>", "</tbody>", "</td>", "</template>", "</textarea>", "</tfoot>", "</th>", "</thead>", "</time>", "</title>", "</tr>", "</track>", "</tt>", "</u>", "</ul>", "</var>", "</video>",
            "</wbr>", "</SCR'+'IPT>", "</!DOCTYPE>", "</A>", "</ABBR>", "</ACRONYM>", "</ADDRESS>", "</APPLET>", "</AREA>", "</ARTICLE>", "</ASIDE>", "</AUDIO>", "</B>", "</BASE>", "</BASEFONT>", "</BDI>", "</BDO>", "</BIG>", "</BLOCKQUOTE>", "</BODY>", "</BR>", "</BUTTON>", "</CANVAS>", "</CAPTION>", "</CENTER>", "</CITE>", "</CODE>", "</COL>", "</COLGROUP>", "</DATALIST>", "</DD>", "</DEL>", "</DETAILS>", "</DFN>", "</DIALOG>", "</DIR>", "</DIV>", "</DL>", "</DT>", "</EM>", "</EMBED>", "</FIELDSET>", "</FIGCAPTION>", "</FIGURE>", "</FONT>", "</FOOTER>", "</FORM>", "</FRAME>", "</FRAMESET>", "</H1>", "</HEAD>", "</HEADER>", "</HR>", "</HTML>", "</I>", "</IFRAME>", "</IMG>", "</INPUT>", "</INS>", "</KBD>", "</LABEL>", "</LEGEND>", "</LI>", "</LINK>", "</MAIN>", "</MAP>", "</MARK>", "</MENU>", "</MENUITEM>", "</META>", "</METER>", "</NAV>", "</NOFRAMES>", "</NOSCRIPT>", "</OBJECT>", "</OL>", "</OPTGROUP>", "</OPTION>", "</OUTPUT>", "</P>", "</PARAM>", "</PICTURE>", "</PRE>", "</PROGRESS>", "</Q>", "</RP>", "</RT>", "</RUBY>", "</S>", "</SAMP>", "</SCRIPT>", "</SECTION>", "</SELECT>", "</SMALL>", "</SOURCE>", "</SPAN>", "</STRIKE>", "</STRONG>", "</STYLE>", "</SUB>", "</SUMMARY>", "</SUP>", "</TABLE>", "</TBODY>", "</TD>", "</TEMPLATE>", "</TEXTAREA>", "</TFOOT>", "</TH>", "</THEAD>", "</TIME>", "</TITLE>", "</TR>", "</TRACK>", "</TT>", "</U>", "</UL>", "</VAR>", "</VIDEO>", "</WBR>"]
start_tags = ["&nbsp;", "<scr'+'ipt>", "<!----!>", "<!DOCTYPE>", "<a>", "<abbr>", "<acronym>", "<address>", "<applet>", "<area>", "<article>", "<aside>", "<audio>", "<b>", "<base>", "<basefont>", "<bdi>", "<bdo>", "<big>", "<blockquote>", "<body>", "<br>", "<button>", "<canvas>", "<caption>", "<center>", "<cite>", "<code>", "<col>", "<colgroup>", "<datalist>", "<dd>", "<del>", "<details>", "<dfn>", "<dialog>", "<dir>", "<div>", "<dl>", "<dt>", "<em>", "<embed>", "<fieldset>", "<figcaption>", "<figure>", "<font>", "<footer>", "<form>", "<frame>", "<frameset>", "<h1>", "<head>", "<header>", "<hr>", "<html>", "<i>", "<iframe>", "<img>", "<input>", "<ins>", "<kbd>", "<label>", "<legend>", "<li>", "<link>", "<main>", "<map>", "<mark>", "<menu>", "<menuitem>", "<meta>", "<meter>", "<nav>", "<noframes>", "<noscript>", "<object>", "<ol>", "<optgroup>", "<option>", "<output>", "<p>", "<param>", "<picture>", "<pre>", "<progress>", "<q>", "<rp>", "<rt>", "<ruby>", "<s>", "<samp>", "<script>", "<section>", "<select>", "<small>", "<source>", "<span>", "<strike>", "<strong>", "<style>", "<sub>", "<summary>", "<sup>", "<table>", "<tbody>", "<td>", "<template>", "<textarea>", "<tfoot>", "<th>", "<thead>", "<time>", "<title>", "<tr>", "<track>", "<tt>", "<u>", "<ul>", "<var>", "<video>",
              "<wbr>", "<SCR'+'IPT>", "<!-->", "<!DOCTYPE>", "<A>", "<ABBR>", "<ACRONYM>", "<ADDRESS>", "<APPLET>", "<AREA>", "<ARTICLE>", "<ASIDE>", "<AUDIO>", "<B>", "<BASE>", "<BASEFONT>", "<BDI>", "<BDO>", "<BIG>", "<BLOCKQUOTE>", "<BODY>", "<BR>", "<BUTTON>", "<CANVAS>", "<CAPTION>", "<CENTER>", "<CITE>", "<CODE>", "<COL>", "<COLGROUP>", "<DATALIST>", "<DD>", "<DEL>", "<DETAILS>", "<DFN>", "<DIALOG>", "<DIR>", "<DIV>", "<DL>", "<DT>", "<EM>", "<EMBED>", "<FIELDSET>", "<FIGCAPTION>", "<FIGURE>", "<FONT>", "<FOOTER>", "<FORM>", "<FRAME>", "<FRAMESET>", "<H1>", "<HEAD>", "<HEADER>", "<HR>", "<HTML>", "<I>", "<IFRAME>", "<IMG>", "<INPUT>", "<INS>", "<KBD>", "<LABEL>", "<LEGEND>", "<LI>", "<LINK>", "<MAIN>", "<MAP>", "<MARK>", "<MENU>", "<MENUITEM>", "<META>", "<METER>", "<NAV>", "<NOFRAMES>", "<NOSCRIPT>", "<OBJECT>", "<OL>", "<OPTGROUP>", "<OPTION>", "<OUTPUT>", "<P>", "<PARAM>", "<PICTURE>", "<PRE>", "<PROGRESS>", "<Q>", "<RP>", "<RT>", "<RUBY>", "<S>", "<SAMP>", "<SCRIPT>", "<SECTION>", "<SELECT>", "<SMALL>", "<SOURCE>", "<SPAN>", "<STRIKE>", "<STRONG>", "<STYLE>", "<SUB>", "<SUMMARY>", "<SUP>", "<TABLE>", "<TBODY>", "<TD>", "<TEMPLATE>", "<TEXTAREA>", "<TFOOT>", "<TH>", "<THEAD>", "<TIME>", "<TITLE>", "<TR>", "<TRACK>", "<TT>", "<U>", "<UL>", "<VAR>", "<VIDEO>", "<WBR>"]

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

options_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weights_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def variable(v, arg_use_cuda=True, volatile=False):
    if use_cuda and arg_use_cuda:
        return Variable(v, volatile=volatile).cuda()
    return Variable(v, volatile=volatile)


def pad_seq(seq, max_len, pad_token=0):
    if len(seq) <max_len:
        seq = seq[:]
        seq += [pad_token for i in range(max_len - len(seq))]
    return seq


def pad_elmo(batch):
    max_len = len(max(batch, key=len))
    padded_batch=[sequence + [[0]*1024 for i in range(max_len - len(sequence))] for sequence in batch]
    return padded_batch

def pad_seq_elmo(seq, max_len,size=1024):
    diff = max_len - len(seq)
    if diff == 0:
        return seq
    padded = np.zeros((diff, size))
    padded_batch=np.concatenate((seq ,padded),axis=0)
    return padded_batch

def view_data_point(data_point, vocab):
    print(" ".join([vocab.get_word(id) for id in data_point.question_tokens]))
    print(" ".join([vocab.get_word(id) for id in data_point.candidates[data_point.answer_indices[0]]]))

def view_span_data_point(data_point, vocab):
    print(" ".join([vocab.get_word(id) for id in data_point.question_tokens]))
    print(" ".join([vocab.get_word(id) for id in data_point.answer_tokens]))

    ansnwer_from_context = data_point.context_tokens[data_point.span_indices[0]:data_point.span_indices[1] + 1]
    print(" ".join([vocab.get_word(id) for id in ansnwer_from_context]))

def get_pretrained_emb(embedding_path, word_to_id, dim):
    word_emb = []
    print("Loading pretrained embeddings from {0}".format(embedding_path))
    for _ in range(len(word_to_id)):
        word_emb.append(np.random.uniform(-math.sqrt(3.0 / dim), math.sqrt(3.0 / dim), size=dim))

    print("length of dict: {0}".format(len(word_to_id)))
    pretrain_word_emb = {}
    for line in codecs.open(embedding_path, "r", "utf-8", errors='replace'):
        items = line.strip().split()
        if len(items) == dim + 1:
            try:
                pretrain_word_emb[items[0]] = np.asarray(items[1:]).astype(np.float32)
            except ValueError:
                continue

    not_covered = 0
    for word, id in word_to_id.iteritems():
        if word in pretrain_word_emb:
            word_emb[id] = pretrain_word_emb[word]
        elif word.lower() in pretrain_word_emb:
            word_emb[id] = pretrain_word_emb[word.lower()]
        else:
            not_covered += 1

    emb = np.array(word_emb, dtype=np.float32)

    print("Word number not covered in pretrain embedding: {0}".format(not_covered))
    return emb