import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Update matplotlib parameters for figure aesthetics
plt.rcParams.update({
    'xtick.labelsize': 26,   # x-axis tick labels
    'ytick.labelsize': 26,   # y-axis tick labels
    'axes.labelsize': 26,     # Axis labels (x and y)
    'lines.linewidth': 3,      # Default line width for plot lines
    'figure.titlesize': 36,       # Font size for figure suptitles
    'axes.titlesize': 32,        # Font size for axes titles
    'legend.fontsize' : 26,    # Legend font size
    'lines.markersize' : 10    # Default marker size
})

sns.set_palette("colorblind")

# Read mesh data from Excel file
file_path = "emimesh_data.xlsx"
df_raw = pd.read_excel(file_path, header=None)

subtables = {}
df_raw.drop(df_raw.columns[:2], axis=1, inplace=True)
mu_rows = df_raw.index[df_raw.iloc[:, 0].astype(str).str.startswith("mu =")].tolist()
N_columns = [10, 25, 50, 100, 200, 300, 400, 500]
N_subset = N_columns[3:]

# Extract data for μ = 5, 10, 20, 30
mu_values = [5, 10, 20, 30]

for idx, mu_start in enumerate(mu_rows):
    mu_end = mu_rows[idx+1] if idx+1 < len(mu_rows) else len(df_raw)
    block = df_raw.iloc[mu_start:mu_end].reset_index(drop=True)
    
    # Extract mu value
    mu_label = str(block.iloc[0,0]).strip()
    mu_value = int(mu_label.split("=")[1])
    if mu_value not in mu_values:
        continue

    # Find the "N" row
    n_row = block[block.iloc[:, 0].astype(str).str.strip() == "N"].index[0]

    # Extract N values (vertical list under "N")
    N_values = block.iloc[n_row, 1:len(N_columns)+1].dropna().astype(int).tolist()

    # Extract metrics and their values
    metrics = []
    values = []
    i = n_row+1
    while i < len(block):
        label = str(block.iloc[i, 0]).strip()
        if label == "" or label.startswith("mu =") or label == "nan":
            # End of this metric block
            break
        metrics.append(label)
        vals = block.iloc[i, 1:len(N_values)+1].astype(float).tolist()
        values.append(vals)
        i += 1

    # Build DataFrame
    df_mu = pd.DataFrame(values, index=metrics, columns=N_values)

    # Keep only data for N in N_subset
    df_mu = df_mu.loc[:, N_subset]

    subtables[int(mu_value)] = df_mu

# Example: print summary
for mu, table in subtables.items():
    print(f"mu = {mu}: {table.shape} (rows x columns)")
    print(table.head(), "\n")
    
colors = {5: 'blue', 10: 'orange', 20: 'green', 30: 'red'}
data: dict[pd.DataFrame] = {mu: table for mu, table in subtables.items()}

# Set figure size
figsize = (14, 10)
# Plot (a): Total # vertices
fig1, ax1 = plt.subplots(figsize=figsize)
for mu in mu_values:
    ax1.plot(data[mu].columns, data[mu].loc['npoints'], label=f'μ = {mu}', color=colors[mu])
ax1.set_yscale('log')
ax1.set_title('Total # vertices')
ax1.set_xlabel('N biological cells')
ax1.set_ylabel('Vertices (log scale)')
ax1.legend()

# Plot (b): Membrane # v / total # v
fig2, ax2 = plt.subplots(figsize=figsize)
for mu in mu_values:
    ratio = [m / t for m, t in zip(data[mu].loc['npoints_membrane'], data[mu].loc['npoints'])]
    ax2.plot(data[mu].columns, ratio, label=f'μ = {mu}', color=colors[mu])
ax2.set_title('Membrane # v / total # v')
ax2.set_xlabel('N biological cells')
ax2.set_ylabel('Ratio')
ax2.legend()

# Plot (c): Percentage ECS volume fraction
fig3, ax3 = plt.subplots(figsize=figsize)
for mu in mu_values:
    percentage = [100 * val for val in data[mu].loc['ecs_share']]
    ax3.plot(data[mu].columns, percentage, label=f'μ = {mu}', color=colors[mu])
ax3.set_title('Percentage ECS volume fraction')
ax3.set_xlabel('N biological cells')
ax3.set_ylabel('% ECS volume')
ax3.legend()

# Plot (d): Computational cells
fig4, ax4 = plt.subplots(figsize=figsize)
for mu in mu_values:
    ax4.plot(data[mu].columns, data[mu].loc['ncompcells'], label=f'μ = {mu}', color=colors[mu])
ax4.set_yscale('log')
ax4.set_title('Computational cells')
ax4.set_xlabel('N biological cells')
ax4.set_ylabel('# Cells')
ax4.legend()

# Show all figures
plt.show()