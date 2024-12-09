import pandas as pd
from scipy import stats


def read(path, suffix):
    df = pd.read_csv(path, sep="\t")
    df["runs"] = pd.to_numeric(df["runs"])
    return df.add_suffix(suffix)
    # num_cols = df.cols
    # num_cols.remove('command')


base = read("loge_baselines.tsv", "_base")
# print(base)
exp = read("loge_benchmarks.tsv", "_exp")
# print(exp)

df = pd.concat([base, exp], axis=1)
for index, row in df.iterrows():
    assert row["command_base"] == row["command_exp"].replace(
        " --loss=loge", "")

# print(df.columns)
# print(df.dtypes)


def t_test(df, metric):
    df[f"{metric}_abs_delta"] = df[f"{metric}_mean_exp"] - df[
        f"{metric}_mean_base"]
    df[f"{metric}_rel_delta"] = (df[f"{metric}_abs_delta"] /
                                 df[f"{metric}_mean_base"])
    if "acc" in metric:
        df[f"{metric}_abs_delta"] = df[f"{metric}_abs_delta"]
    df["t_test_tmp"] = df.apply(
        lambda row: stats.ttest_ind_from_stats(
            row[f"{metric}_mean_base"],
            row[f"{metric}_std_base"],
            row["runs_base"],
            row[f"{metric}_mean_exp"],
            row[f"{metric}_std_exp"],
            row["runs_exp"],
            equal_var=False,
        ),
        axis=1,
    )
    df[f"{metric}_tstat"] = df["t_test_tmp"].map(lambda tout: tout.statistic)
    df[f"{metric}_pval"] = df["t_test_tmp"].map(lambda tout: tout.pvalue)
    return df.drop(columns=["t_test_tmp"])


df = t_test(df, "val_acc")
df = t_test(df, "test_acc")
df = t_test(df, "duration")

df = df.rename(columns={
    "command_base": "nll_command"
}).set_index("nll_command")

df.to_csv("loge_t_test_out.tsv", sep='\t')

cols = [
    "val_acc_abs_delta",
    "test_acc_abs_delta",
    # "test_acc_pval",
    "duration_rel_delta",
    # "val_acc_abs_delta",
    # "val_acc_pval",
    # "test_acc_mean_base",
    # "test_acc_mean_exp",
    # "duration_pval",
]
for col in cols:
    df[col] = df[col].map(lambda x: f"{x:.2%}")

print(df.query("test_acc_pval < .05")[cols].to_markdown())
