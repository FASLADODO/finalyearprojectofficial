import csv

f = open("windsizthreshresults.txt","rw+") 
thresholds=[200, 150, 100, 50, 25, 10, 0]
sizes=[100,200,300,400,500]
windows=[1,2,3,5,7,10,15,20,50]
lines=[]
while True:
	temp=f.readline()
	if not temp: break
    	lines.append(temp)
print lines[0]
print lines[1]
print lines[2]
i=1
with open('resultParams.txt', 'wb') as csvfile:
	csvfile.truncate()
	#spamwriter = csv.writer(csvfile, delimiter=' ')
	for t in thresholds:
		print t
		#spamwriter.writerow('Threshold'+ str(t)+", Window, Seen")
		csvfile.write('Threshold'+ str(t)+" Windows"+"\n")
		sizeList = ','.join(map(str, windows)) 
		#spamwriter.writerow("Size," + sizeList)
		csvfile.write("Size," + sizeList+"\n")
		seenAmount=""
		for s in sizes:
			print s
			rowEntries=[]
			for w in windows:
				print w
				i=i+1
				final=0

				for u in range(7):
					l=lines[i]#f.readline()
					i=i+2
					print u
					print l
					keep=l.split("%  (")
					keepy= keep[1].split("/ ")
					final+=int(keepy[0])
					seenAmount=lines[i]
					i=i+1
				rowEntries.append(str(final))
				i=i+1
			resultList=','.join( rowEntries) 
    			#spamwriter.writerow(str(s)+","+resultList)
			csvfile.write(str(s)+","+resultList+"\n")
		#spamwriter.writerow(seenAmount)
		#spamwriter.writerow("")
		csvfile.write(seenAmount+"\n")
		csvfile.write("\n")
			
	
