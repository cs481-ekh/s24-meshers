# Simple implementation for demo skips copies to locals, and skips range checking
#     def constructor(self, Lh, x, domArea, hasConstNullspace, verbose):
#         polyDim = (self.polyDegT+1)*(self.polyDegT+2)/2
#         minStencil = math.ceil(max(1.5*polyDim,3));  # Heuristic
#         computeDomArea = true
#         self.hasConstNullspace = false
#         verbose = false
#         rbfOrderT = 0
#         rbf = polyHarmonic
#         interpMethod = 0
#
#         N = x.shape[0] # get number of rows ( shape gives dimensions, then access row dim)
#         p = math.floor(math.log(N / self.Nmin) / math.log(self.coarseningFactor)) # #Compute number of levels
#
#         kdtree = cKDTree(x) # Build a cKDTree
#         distances, indices = kdtree.query(x, k=2) # Perform nearest neighbor search
#         domArea = x.shape[0] * np.mean(distances[:, 1])**2  # Compute domain area
#         self.domainVol = domArea
#
#         xc = np.empty(p + 1, dtype=object)
#         Nc = np.zeros(p + 1)
#         Nc[0] = N
#         xc[0] = x
#
#         #Build coarsened levels
#         for j in range(2, p + 2):
#             Nc[j] = math.floor(N / coarseningFactor ** (j - 1))
#             xc[j] = PcCoarsen2D(xc[j - 1], Nc[j], domArea)  # starting a j=2, this fills in the next level (first level keep same)
#             Nc[j] = xc[j].shape[0]
#
#         levelsData = []
#
#         # Initialize level data for each level
#         for _ in range(p + 1):
#             level_data = {
#                 'nodes': [],
#                 'stencilSize': self.stencilSizeT,
#                 'rbfOrder': self.rbfOrderT,
#                 'rbfPolyDeg': self.polyDegT,
#                 'rbf': self.rbf,
#                 'idx': [],
#                 'Lh': [],
#                 'DLh': [],
#                 'I': [],
#                 'R': [],
#                 'Mhf': [],
#                 'Nhf': [],
#                 'Mhb': [],
#                 'Nhb': [],
#                 'preSmooth': self.preSmooth,
#                 'postSmooth': self.postSmooth,
#                 'Ihat': 1,
#                 'Rhat': 1,
#                 'w': [],
#                 'Qh': 0
#             }
#             levelsData.append(level_data)
#
#         levelsData[1]['nodes'] = x
#         levelsData[0]['Lh'] = Lh
#         levelsData[0]['w'] = np.ones(N)
#
#
#         obj = {}
#         obj['coarseningFactor'] = 4
#         obj['levelsData'] = levelsData
#
#         return obj