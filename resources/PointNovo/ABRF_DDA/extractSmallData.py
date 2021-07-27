

i=0
with open("spectrums.mgf") as in_f, open("spectrums_small.mgf", 'w') as out_f:
	for line in in_f:
		if line.startswith("BEGIN IONS"):
			i=i+1
		if(i<1500):
			out_f.write(line)
