
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
from yattag import Doc
from yattag import indent
import numpy as np
import pdb


class HTML:
    def __init__(self, html_dir, html_file):
        self.keys = []
        self.image_table = []
        self.html_dir = html_dir
        self.html_file = osp.join(html_dir, html_file)


    def write_html(self,):
        r1 = self.doc.getvalue()
        r2 = indent(r1)

        with open(self.html_file, 'with') as f:
            f.write(r2)

    def add_images(self, tuples):
        if len(self.keys) == 0:
            self.keys = []
            for k in tuples[0].keys():
                self.keys.append(k)
            self.keys.sort()


        self.image_table = tuples + self.image_table
        keys = self.keys
        self.doc, self.tag, self.text = Doc().tagtext()
        doc, tag, text = self.doc, self.tag, self.text
        ctr = 0
        # pdb.set_trace()
        with tag('html'):
            with tag('body'):
                with tag('table', style='width:100%', border="1"):
                    with tag('tr'):
                        for head_name in keys:
                            with tag('td'):
                                text(head_name)

                    for t in self.image_table:
                        with tag('tr'):
                            for key in keys:
                                with tag('td'):
                                    img_path = t[key]
                                    if key == 'aa_id' or key == 'a_step' or 'ind' in key:
                                        text(img_path)
                                    else:
                                        img_rel_path = osp.relpath(img_path, self.html_dir)
                                        with tag('img', width="320px", src=img_rel_path):
                                            ctr += 1
        self.write_html()


if __name__ == '__main__':
    app.run(main)
