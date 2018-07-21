import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#import peakutils

class casa_data():
	'''
	Class for CasaXPS data. Reads exported ASCII file and provides convenient plotting function
	Arguments:
	----------
	file: ASCII data file to read
	'''
	def __init__(self,file):
		self.data = pd.read_csv(file,skiprows=6,sep='\t')
		#remove extra columns (due to extraneous tabs in header row)
		unnamed = [c for c in self.data.columns if c[0:8]=='Unnamed:']
		self.data.drop(unnamed,axis=1,inplace=True)
		self.cycles = [c for c in self.data.columns if c[0:5]=='Cycle']
		if 'Background' in self.data.columns:
			compstart = list(self.data.columns).index(self.cycles[-1]) + 1
			compend = list(self.data.columns).index('Background')
			self.components = self.data.columns[compstart:compend]
		else:
			self.components = []
		self.data.rename({'B.E.':'BE'},axis=1,inplace=True)
		
	@property
	def peaks(self):
		'''
		Dict of peak positions. Dict item format is component:(xcoord,ycoord)
		'''
		peaks = {}
		for comp in self.components:
			idx = self.data[comp].idxmax()
			peaks[comp] = (self.data['BE'][idx],self.data[comp][idx])
		return peaks
		
	def rename(self, components, cycles=None):
		'''
		Rename columns in dataframe. 
		Arguments:
		----------
		components: list of names to assign to components (peak fits)
		cycles: list of names to assign to cycles (measured data). If None, assigns numeric names (Cycle1, Cycle2, ...)
		'''
		rename = {}
		if cycles==None:
			cycles = []
			for i,c in enumerate(self.cycles):
					cyc = 'Cycle{}'.format(i)
					rename[c] = cyc
					cycles.append(cyc)
		else:
			for i,c in enumerate(self.cycles):
				rename[c] = cycles[i]
		
		for c,cname in zip(self.components,components):
			rename[c] = cname
		self.data.rename(rename,axis=1,inplace=True)
		self.cycles = cycles
		self.components = components
	
	def plot(self,title=None,peaklabels=None,labeloffset=(0.1,0.1),lw=2,cyclw=1,fs=12,fontweight='bold',xint=5,xmin=None,yticks=False):
		'''
		Plot all cycles, components, background, and envelope. Automatically label peaks
		Arguments:
		----------
		title: plot title (placed in upper right corner)
		peaklabels: list of peak labels. If None, uses dataframe component column names
		labeloffset: label offset from peak (in fractional axes coords)
		lw: linewidth for components, background, and envelope. Defaults to 2
		cyclw: linewidth for cycles. Defaults to 1
		fs: fontsize for axes labels, peak labels, and title. Tick labels are 2 points smaller. Defaults to 12
		fontweight: fontweight for all text
		xint: x tick interval. Defaults to 5
		xmin: x tick start. Defaults to lowest integer within xlim
		yticks: If True, show y ticks and labels in cnt/s. If False, don't show y ticks
		'''
		fig, ax = plt.subplots()
		for cyc in self.cycles:
			ax.plot(self.data['BE'],self.data[cyc],color='darkgray',lw=cyclw)
		for i,comp in enumerate(self.components):
			ax.plot(self.data['BE'],self.data[comp],lw=lw)
			#label peaks
			idx = self.data[comp].idxmax() #peakutils.indexes(self.data[comp],thres=0.9)[0]
			xval = self.data['BE'][idx]
			yval = self.data[comp][idx]
			xrng = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
			yrng = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
			xoffset = labeloffset[0]*xrng
			yoffset = labeloffset[1]*yrng
			if peaklabels==None:
				ax.annotate(comp,xy=(xval+xoffset/5,yval+yoffset/5),xytext=(xval + xoffset,yval + yoffset),
							horizontalalignment='center',arrowprops={'arrowstyle':'-'},fontsize=fs,fontweight='bold')
			else:
				ax.annotate(peaklabels[i],xy=(xval+xoffset/5,yval+yoffset/5),xytext=(xval + xoffset,yval + yoffset),
							horizontalalignment='center',arrowprops={'arrowstyle':'-'},fontsize=fs,fontweight='bold')
		if len(self.components) > 0:
			ax.plot(self.data['BE'],self.data['Background'],color='gray',lw=lw)
			ax.plot(self.data['BE'],self.data['Envelope'],color='red',lw=lw)
		
		ax.set_xlim([self.data['BE'].min(),self.data['BE'].max()])
		ax.invert_xaxis()
		
		ax.set_xlabel('Binding Energy (eV)',fontsize=fs,fontweight='bold')
		
		#tick formatting
		tickfd = {'fontweight':fontweight,'fontsize':fs-2}
		if xmin==None:
			xmin = np.ceil(ax.get_xlim()[1])
		xmax = np.floor(ax.get_xlim()[0])
		xticks = np.arange(xmin,xmax+0.1,xint)
		ax.set_xticks(xticks)
		ax.set_xticklabels(ax.get_xticks(),fontdict=tickfd)
		if xint%1 == 0:
			ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
		else:
			ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		
		
		if yticks==True:
			ax.set_yticklabels(ax.get_yticks(),fontdict=tickfd)
			ax.set_ylabel('Intensity (cnt/s)',fontsize=fs,fontweight='bold')
			ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
		else:
			ax.set_yticks([])
			ax.set_ylabel('Intensity (a.u.)',fontsize=fs,fontweight='bold')
		
		if title is not None:
			ax.set_title(title, fontsize=fs, fontweight=fontweight,x=0.92,y=0.9)
		
		return fig, ax
				