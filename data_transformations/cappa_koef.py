#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
from endoanalysis.similarity import KPSimilarity
from endoanalysis.agreement import compute_kappas, load_agreement_keypoints
from endoanalysis.agreement import plot_agreement_matrices, compute_icc, plot_numbers, get_keypoints_nums
from endoanalysis.agreement import ptg_agreement, stud_agreement, ptg_stud_agreement


YMLS = [
    "../test_cappa_koef/test_plasmatic.yaml",  
]
STUDIES = ["test", "not test"]
EXPERTS =  ['anna_t', 'alina_m']


SCALE = 6

similarity = KPSimilarity(scale=SCALE)

DROP_MISSED = False

experts_mapping = {expert: expert_i for expert_i, expert in enumerate(EXPERTS)}
kappas = {}
deltas = {}

for study_name, yml_path in zip(STUDIES, YMLS):
    print (study_name)
    print (yml_path)
    keypoints = load_agreement_keypoints(yml_path)
    kappas_study, experts_to_ids = compute_kappas(
        keypoints,
        similarity,
        EXPERTS,
        drop_missed=DROP_MISSED
    )
    kappas[study_name] = kappas_study
    
print("Kappas")
print (kappas)
fig, ax = plot_agreement_matrices(kappas, STUDIES, EXPERTS, fig_size=(24, 6))
# fig.set_facecolor("white")
# fig.savefig("figs/kappas_drop", dpi=300, bbox_inches='tight',facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

