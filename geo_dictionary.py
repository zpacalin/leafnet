import numpy as np
import random
import pandas as pd
import csv

def buildDict():
    """
    d["species name"] = [top(lat), bottom(lat), left(lon), right(lon)]
    if non-contiguous regions, multiple rows
    """
    d = {}
    #populate 
    d["Abies concolor"] = [48.534670, 31.698416, -124.759017, -103.137923]
    d["Abies nordmanniana"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Acer campestre"] = [42.137475, 32.887122, -124.319564, -114.212142] #,[45.009850, 36.754713, -88.375283, -70.357705]]
    d["Acer ginnala"] = [49.006870, 36.749780, -104.040682, -66.427822]
    d["Acer griseum"] = [53.478821, 32.039037, -128.985915, -73.767240]
    d["Acer negundo"] = [49.187001, 25.529928, -122.963466, -66.098232]
    d["Acer palmatum"] = [56.901284, 38.934589, -96.307116, -71.513381]
    d["Acer pensylvanicum"] = [63.172186, 31.237182, -98.341995, -57.912307]
    d["Acer platanoides"] = [59.955603,42.286904, -139.991113, -104.073444] #,[62.025851, 35.753991, -97.100246, -54.209621]]
    d["Acer pseudoplatanus"] = [56.661291, 33.145267, -97.451808, -64.932277] #,[59.981616, 49.502659, -139.815090, -110.108058]]
    d["Acer rubrum"] = [62.758944, 25.966330, -111.690090, -52.803371]
    d["Acer saccharinum"] = [58.544404, 27.067332, -109.053371, -58.604152]
    d["Acer saccharum"] = [61.604333, 25.125829, -116.168630, -63.412758]
    d["Aesculus flava"] = [45.795822, 29.340940, -93.764223, -71.264223]
    d["Aesculus glabra"] = [52.507485, 27.049269, -105.102113, -65.827486]
    d["Aesculus hippocastamon"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Aesculus pavi"] = [43.444884, 25.631548, -105.102113, -72.331392]
    d["Ailanthus altissima"] = [60.705409,25.383662,-125.549166,-58.576510]
    d["Albizia julibrissin"] = [45.259365, 25.224747, -124.494478, -67.541353]
    d["Amelanchier arborea"] = [61.217340, 24.746758, -107.092135, -58.928072]
    d["Amelanchier canadensis"] = [60.962404, 27.586126, -95.139010, -56.642916]
    d["Amelanchier laevis"] = [60.532928, 28.825354, -97.599947, -54.006197]
    d["Asimina triloba"] = [56.108765, 25.542368, -107.092135, -70.529635]
    d["Betula alleghaniensis"] = [60.962404, 29.745231, -97.951510, -51.721041]
    d["Betula jacqemontii"] = [42.954195, 41.325041, -73.750299, -69.883111]
    d["Betula lenta"] = [56.754066, 30.299133, -97.063287, -70.344537]
    d["Betula nigra"] = [48.808477, 25.275911, -103.918756, -69.641412]
    d["Betula populifolia"] = [60.501732, 31.205501, -100.227349, -58.039849]
    d["Broussonettia papyrifera"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Carpinus betulus"] = [44.089345, 35.605711, -86.340631, -69.289849]
    d["Carpinus caroliniana"] = [59.446321, 26.275911, -108.840631, -57.512506]
    d["Carya cordiformis"] = [59.890166, 25.167391, -107.785943, -55.227349]
    d["Carya glabra"] = [56.171387, 25.167391, -106.555474, -68.641412]
    d["Carya ovata"] = [59.890166, 26.904662, -107.961724, -64.016412]
    d["Carya tomentosa"] = [47.041852, 26.590718, -105.852349, -64.367974]
    d["Castanea dentata"] = [57.232829, 25.044448, -93.547662, -67.493241]
    d["Catalpa bignonioides"] = [48.834821, 26.055330, -125.061601, -67.157991]
    d["Catalpa speciosa"] = [49.281085, 25.501325, -115.322053, -67.245881]
    d["Cedrus atlantica"] =  [41.757843, 32.234702, -123.097243, -74.230055]
    d["Cedrus deodara"] = [34.902627, 30.314592, -86.054475, -75.243928]
    d["Cedrus libani"] = [49.655940, 33.489701, -119.142165, -72.032790]
    d["Celtis occidentalis"] = [61.105316, 24.695465, -118.881624, -57.885530]
    d["Celtis tenuifolia"] = [56.757859, 25.332635, -110.444124, -65.444124]
    d["Cercidiphyllum japonicum"] = [47.046567, 39.511269, -82.494905, -67.026155]
    d["Cercis canadensis"] = [57.236574, 25.332635, -111.674592, -64.916780]
    d["Chamaecyparis pisifera"] = [41.483358, 39.172107, -75.539802, -73.430427]
    d["Chamaecyparis thyoides"] = [45.678823, 24.890774, -90.195564, -67.344002]
    d["Chionanthus retusus"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Chionanthus virginicus"] = [44.437194, 25.606214, -103.379158, -65.498298]
    d["Cladrastis lutea"] = [45.432654, 29.424625, -103.554939, -67.168220]
    d["Cornus florida"] = [56.576315, 25.827395, -107.690291, -65.502791]
    d["Cornus kousa"] = [46.337682, 39.391895, -78.334823, -69.897323]
    d["Cornus mas"] = [45.677639, 40.915845, -80.359975, -73.108998]
    d["Corylus colurna"] = [49.826335, 29.366440, -120.636305, -74.845290]
    d["Crataegus crus-galli"] = [60.472279, 24.953998, -112.330190, -55.552846]
    d["Crataegus laevigata"] = [61.073067, 34.983031, -98.267690, -57.134877]
    d["Crataegus phaenopyrum"] = [48.884809, 25.113266, -97.916127, -63.990346]
    d["Crataegus pruinosa"] = [62.408538, 30.007140, -107.351165, -57.646424]
    d["Crataegus viridis"] = [40.592982, 26.445846, -104.931580, -66.962830]
    d["Cryptomeria japonica"] = [39.380899, 29.854803, -97.372987, -73.466737]
    d["Diospyros virginiana"] = [46.327858, 26.603125, -124.970643, -67.665955]
    d["Eucommia ulmoides"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Evodia daniellii"] = [46.447880, 31.815253, -123.952786, -75.788724]
    d["Fagus grandifolia"] = [61.863848, 26.288351, -116.884705, -52.724549]
    d["Ficus carica"] = [45.100560, 26.917034, -124.443299, -65.205018]
    d["Fraxinus americana"] = [60.508563, 25.020766, -113.544862, -57.294862]
    d["Fraxinus nigra"] = [62.111520, 33.298266, -110.029237, -54.306580]
    d["Fraxinus pennsylvanica"] = [61.023634, 24.061371, -123.037049, -55.361268]
    d["Ginkgo biloba"] = [44.727113, 30.584948, -89.638612, -72.745846]
    d["Gleditsia triacanthos"] = [55.334759, 26.518737, -130.116452, -65.956295]
    d["Gymnocladus dioicus"] = [56.322011, 24.775747, -105.331295, -65.780514]
    d["Halesia tetraptera"] = [48.055308, 26.675916, -106.913327, -64.550045]
    d["Ilex opaca"] = [45.651668, 25.174107, -106.210202, -68.106966]
    d["Juglans cinerea"] = [60.844367, 25.957042, -100.934114, -56.109895]
    d["Juglans nigra"] = [62.754215, 25.723990, -117.457552, -54.703645]
    d["Juniperus virginiana"] = [62.754215, 25.723990, -117.457552, -54.703645]
    d["Koelreuteria paniculata"] = [49.267077, 26.114984, -114.293489, -64.195833]
    d["Larix decidua"] = [57.040123, 33.577085, -101.988802, -58.922395]
    d["Liquidambar styraciflua"] = [45.828022, 24.857309, -108.844270, -66.406054]
    d["Liriodendron tulipifera"] = [55.727591, 24.767559, -104.594530, -64.340624]
    d["Maclura pomifera"] = [49.440112, 25.245468, -124.633593, -65.043749]
    d["Magnolia acuminata"] = [57.088979, 24.927069, -104.242968, -65.395312]
    d["Magnolia denudata"] = [54.662487, 27.625524, -130.908223, -70.734661]
    d["Magnolia grandiflora"] = [41.706366, 26.037809, -104.594530, -68.559374]
    d["Magnolia macrophylla"] = [44.025035, 29.612412, -92.992968, -68.910937]
    d["Magnolia soulangiana"] = [57.088979, 38.480063, -98.266405, -76.117968]
    d["Magnolia stellata"] = [41.837464, 38.389702, -84.731249, -80.490526]
    d["Magnolia tripetala"] = [45.810252, 25.764456, -100.705370, -68.350750]
    d["Magnolia virginiana"] = [43.168472, 25.288590, -107.989422, -72.217937]
    d["Malus angustifolia"] = [42.975862, 25.288590, -107.989422, -72.217937]
    d["Malus baccata"] = [60.483293, 32.383513, -97.411889, -52.499780]
    d["Malus coronaria"] = [56.216380, 30.038812, -110.433912, -73.926254]
    d["Malus floribunda"] = [48.713122, 28.811417, -123.800821, -65.509916]
    d["Malus hupehensis"] = [48.919396, 46.373996, -123.800821, -118.363025]
    d["Malus pumila"] = [58.941905, 25.071371, -137.391345, -52.664783]
    d["Metasequoia glyptostroboides"] = [45.387419, 38.878805, -84.305408, -72.703845]
    d["Morus alba"] = [60.483293, 25.383513, -97.411889, -56.499780]
    d["Morus rubra"] = [56.405211, 24.445275, -107.870456, -67.586956]
    d["Nyssa sylvatica"] = [56.405211, 24.445275, -107.870456, -67.586956]
    d["Ostrya virginiana"] = [62.495451, 25.291140, -113.479349, -55.998880]
    d["Oxydendrum arboreum"] = [39.039017, 30.385423, -89.235990, -76.755521] 
    d["Paulownia tomentosa"] = [45.311016, 25.608587, -107.854349, -60.744974]
    d["Phellodendron amurense"] = [60.741309, 33.400948, -94.494974, -57.229349]
    d["Picea abies"] = [61.786429, 35.053036, -99.065286, -54.592630]
    d["Picea orientalis"] = [47.704763, 29.700688, -123.689115, -78.601224]
    d["Picea pungens"] = [49.171371, 33.747741, -117.170755, -103.811380]
    d["Pinus bungeana"] = [52.696897, 36.457347, -126.648370, -65.652277]
    d["Pinus cembra"] = [43.581031, 22.594542, -123.484308, -73.210870]
    d["Pinus densiflora"] = [52.696897, 36.457347, -126.648370, -65.652277]
    d["Pinus echinata"] = [44.968582, 25.071289, -106.034147, -73.330516]
    d["Pinus flexilis"] = [43.708233, 38.548857, -118.210870, -110.300714]
    d["Pinus koraiensis"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Pinus nigra"] = [61.523117, 43.453561, -138.953058, -63.718683]
    d["Pinus parviflora"] = [50.848131, 36.032047, -124.187433, -64.421808]
    d["Pinus peucea"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Pinus pungens"] = [40.714626, 35.889766, -80.945245, -79.187433]
    d["Pinus resinosa"] = [51.836324, 41.771971, -90.261652, -60.554620]
    d["Pinus rigida"] = [49.951789, 35.030720, -82.527277, -72.507745]
    d["Pinus strobus"] = [54.877115, 40.179549, -93.601495, -69.343683]
    d["Pinus sylvestris"] = [59.356047, 45.089660, -136.140558, -67.410089]
    d["Pinus taeda"] = [39.096649, 29.993768, -95.886652, -77.429620]
    d["Pinus thunbergii"] = [42.164059, 30.902983, -124.714777, -73.386652]
    d["Pinus virginiana"] = [44.965424, 34.886656, -81.472589, -75.847589]
    d["Pinus wallichiana"] = [42.940987, 33.725075, -124.714777, -77.605402]
    d["Platanus acerifolia"] = [42.960936, 35.196807, -124.553567, -115.764505]
    d["Platanus occidentalis"] = [58.554278, 25.349600, -105.569192, -62.854349]
    d["Populus deltoides"] = [60.340875, 25.349600, -140.901224, -57.229349]
    d["Populus grandidentata"] = [59.903051, 33.308140, -140.022317, -57.053567]
    d["Populus tremuloides"] = [70.340875, 25.349600, -167.901224, -57.229349]
    d["Prunus pensylvanica"] = [61.223087, 29.446790, -130.383162, -60.437947]
    d["Prunus sargentii"] = [49.790655, 34.246537, -123.073880, -76.140286]
    d["Prunus serotina"] = [35.735381, 26.811832, -106.316853, -91.287556]
    d["Prunus serrulata"] = [42.346891, 35.376194, -124.461131, -116.287303]
    d["Prunus subhirtella"] = [42.411816, 39.083724, -84.383006, -80.867381]
    d["Prunus virginiana"] = [56.046312, 25.175083, -131.580272, -59.534886]
    d["Prunus yedoensis"] = [45.223249, 32.781030, -114.636380, -78.777005] 
    d["Pseudolarix amabilis"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Ptelea trifoliata"] = [60.931195, 25.325586, -115.486261, -56.072198]
    d["Pyrus calleryana"] = [43.962321, 25.642943, -107.927667, -62.927667]
    d["Quercus acutissima"] = [41.509754, 30.298374, -94.040948, -68.376886]
    d["Quercus alba"] = [60.931195, 27.216953, -112.673761, -58.357355]
    d["Quercus bicolor"] = [60.673948, 29.689417, -106.345636, -59.939386]
    d["Quercus cerris"] = [44.340689, 39.910941, -76.814386, -69.080011]
    d["Quercus coccinea"] = [45.584389, 31.204749, -100.369073, -68.728448]
    d["Quercus falcata"] = [44.088713, 26.903878, -105.115167, -67.146417]
    d["Quercus imbricaria"] = [43.962321, 31.054280, -99.138605, -70.310480]
    d["Quercus macrocarpa"] = [60.587739, 27.060525, -121.462823, -65.740167]
    d["Quercus marilandica"] = [45.337806, 25.060525, -106.697198, -63.740167]
    d["Quercus michauxii"] = [41.245954, 26.903878, -104.939386, -66.970636]
    d["Quercus montana"] = [44.214836, 29.689417, -91.228448, -68.904230]
    d["Quercus muehlenbergii"] = [56.463358, 26.275123, -111.794855, -67.322198]
    d["Quercus nigra"] = [40.848249, 27.684919, -104.763605, -65.037042]
    d["Quercus palustris"] = [57.798781, 30.752628, -108.806573, -67.146417]
    d["Quercus phellos"] = [42.034141, 27.995789, -101.072198, -67.497980]
    d["Quercus robur"] = [57.232353, 37.021352, -94.216730, -63.455011]
    d["Quercus rubra"] = [61.101548, 29.689417, -106.345636, -63.455011]
    d["Quercus shumardii"] = [57.041584, 26.589932, -109.333917, -69.431573]
    d["Quercus stellata"] = [43.326320, 26.275123, -105.818292, -68.201105]
    d["Quercus velutina"] = [57.137092, 26.116780, -105.115167, -64.421808]
    d["Quercus virginiana"] = [37.300979, 26.589318, -103.445245, -74.617120]
    d["Robinia pseudo-acacia"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Salix babylonica"] = [40.581256, 26.116780, -94.656183, -70.222589]
    d["Salix caroliniana"] = [40.313717, 25.642324, -102.038995, -69.695245]
    d["Salix matsudana"] = [42.033631, 37.161021, -110.476495, -97.117120]
    d["Salix nigra"] = [61.773541, 25.958840, -108.191339, -64.070245]
    d["Sassafras albidum"] = [56.268252, 24.527939, -103.621027, -66.882745]
    d["Staphylea trifolia"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Stewartia pseudocamellia"] = [39.311554, 29.471395, -97.146146, -79.040677]
    d["Styrax japonica"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Styrax obassia"] = [48.467997, 26.207927, -125.095365, -68.054349]#
    d["Syringa reticulata"] = [43.835165, 39.504723, -112.937433, -105.027277]
    d["Taxodium distichum"] = [42.683085, 25.958840, -104.148370, -73.210870]
    d["Tilia americana"] = [48.691544, 25.165974, -108.542902, -65.476495]
    d["Tilia cordata"] = [57.610581, 37.719290, -96.765558, -68.464777]
    d["Tilia europaea"] = [59.977448, 34.307874, -97.820245, -65.300714]
    d["Tilia tomentosa"] = [57.326999, 45.337324, -99.050714, -78.660089]
    d["Toona sinensis"] = [48.467997, 26.207927, -125.095365, -68.054349]
    d["Tsuga canadensis"] = [60.500961, 31.354392, -101.160089, -59.148370]
    d["Ulmus americana"] = [60.152882, 26.116780, -119.968683, -59.499933]
    d["Ulmus glabra"] = [58.263752, 31.654134, -107.136652, -66.882745]
    d["Ulmus parvifolia"] = [41.113135, 27.839858, -123.308527, -74.089777]
    d["Ulmus procera"] = [46.680201, 27.216343, -124.714777, -64.597589]
    d["Ulmus pumila"] = [60.414289, 25.483749, -140.183527, -58.093683]
    d["Ulmus rubra"] = [60.414289, 25.165974, -111.531183, -55.984308]
    d["Zelkova serrata"] = [45.951765, 31.803644, -81.472589, -72.683527]

    return d
    
