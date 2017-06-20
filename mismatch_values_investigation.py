
import csv

asdf = set()
plot_ids = set()
with open('2016.csv', 'r') as f:
  lines = list(csv.reader(f))
  lines = lines[1:]

for line in lines:
  for i, value in enumerate(line):
    if ' && ' in value:
      asdf.add(i)
      if i == 3:
        print(value.split(' && '))
        plot_ids.update(value.split(' && '))

print(sorted(asdf))
print(sorted(plot_ids))

plot_id_counts = dict()
for plot_id in plot_ids:
  plot_id_counts[plot_id] = []


for line in lines:
  line_plot_ids = line[3].split(' && ')
  for line_plot_id in line_plot_ids:
    if line_plot_id in plot_id_counts:
      plot_id_counts[line_plot_id].append(line[3])

for k, v in plot_id_counts.items():
  if len(v) == 1:
    continue
  print(k)



plot_plan_tags = set()
plot_plan_cons = set()
plot_plan_ends = set()
for line in lines:
  plot_plan_tags.add(line[27])
  plot_plan_cons.add(line[28])
  plot_plan_ends.add(line[30])

print(sorted(plot_plan_tags))
print(sorted(plot_plan_cons))
print(sorted(plot_plan_ends))
