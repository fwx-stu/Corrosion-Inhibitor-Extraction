# Figure captions

**Figure 5.** Evidence strategies E0–E5 with GLM-4-Plus on dev37 at a record-matching threshold of 0.38. (a) Field Micro F1, (b) record-matching F1, and (c) Null Accuracy. Points and printed values show arithmetic means across nine runs (three prompt variants × three repetitions). Error bars are two-sided 95% Student t confidence intervals across runs (df = 8), conditional on the same set of 37 papers. The dashed line in each panel marks the E0 mean. These intervals describe variation across runs and do not quantify uncertainty from sampling papers.

**Figure 6.** Model capacity and evidence strategy on test50 at a record-matching threshold of 0.38. (a) Field Micro F1, (b) record-matching F1, and (c) Null Accuracy for Qwen2.5-3B, -7B, and -14B under Q0–Q2. (d) Paired differences between Qwen2.5-14B Q1 and GLM-4-Plus Q0; error bars show 95% percentile confidence intervals from 20,000 paired paper-level bootstrap resamples. Positive differences favor Qwen2.5-14B Q1.

**Figure 7.** Strategy differences across document-structure quartiles on test50 at a record-matching threshold of 0.38. (a, b) Record-matching F1 differences and (c, d) Field Micro F1 differences, stratified by document length and table density. The displayed contrasts are Q2−Q0 for Qwen2.5-3B and -7B, and Q1−Q0 for Qwen2.5-14B. Quartiles run from lowest to highest and contain 13, 12, 12, and 13 papers, respectively. Differences are in percentage points (pp); dashed lines indicate zero difference.

**Figure 8.** Field-level (a) precision, (b) recall, and (c) F1 for Qwen2.5-14B under Q0 and Q1 on test50 at a record-matching threshold of 0.38. All 13 schema fields are shown in the same order. Straight segments connect categorical fields to aid visual comparison.

**Figure S1.** Field-level F1 differences between E4 and E0 with GLM-4-Plus on dev37 at a record-matching threshold of 0.38. Bars show the mean of the nine paired run differences across three prompt variants and three repetitions. Positive values favor E4; negative values are shown in orange. Differences are expressed on the original F1 scale, and labels are rounded to three decimal places. All 13 schema fields are shown.

**Figure S2.** Prompt sensitivity on dev37 at a record-matching threshold of 0.38. The 27 combinations of model capacity, evidence strategy, and prompt variant are shown for (a) Field Micro F1, (b) record-matching F1, and (c) Null Accuracy. Cells report pooled scores, and all three heatmaps use the same 0–1 color scale.

**Figure S3.** Paired differences on test50 at a record-matching threshold of 0.38. (a) Selected within-model strategy contrasts and (b) adjacent model-capacity contrasts under Q0. Points show differences in pooled Field Micro F1 and record-matching F1; error bars show 95% percentile confidence intervals from 10,000 paired paper-level bootstrap resamples. Dashed lines indicate zero difference. Holm-adjusted tests are reported separately in the statistical results.

**Figure S4.** Field-level strategy comparison on test50 at a record-matching threshold of 0.38. (a) F1 for Qwen2.5-14B Q0–Q2 and GLM-4-Plus Q0 across the 13 schema fields. (b) Field F1 differences for Qwen2.5-14B Q1−Q0 and Q2−Q0, expressed in percentage points (pp). The dashed line indicates zero difference.
