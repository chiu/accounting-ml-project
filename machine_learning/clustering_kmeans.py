'''
	Attempt to make use of clustering.
'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType

from pyspark.ml import clustering

spark = SparkSession.builder.appName('733').getOrCreate()


annual_compustat = spark.read.csv('/user/vcs/annual_compustat.csv', header=True, inferSchema=True)
ibes_complete = spark.read.csv('/user/vcs/ibes.csv', header=True, inferSchema=True)
dgls = spark.read.csv('/user/vcs/dgls_integrated.csv', header=True, inferSchema=True)

ls = ['CONM', 'ACCHG', 'ACO', 'ACOMINC', 'ACQCSHI', 'ACQGDWL', 'ACQINTAN', 'ACQLNTAL', 'ACT', 'AMGW','AQ','AP','AQC','ARC','AT','AU','AUOP','AUOPIC','BKVLPS','CAPX','CDVC','CEQ','CGA','CHE','CHECH','CI','CIMII','CLD2','CLD3','CLD4','CLD5','CLT','COGS','CSHI','CURNCD','DCOM','DCPSTK','DCVT','DEPC','DLTT','DM','DO','DP','DPACLS','DRC','DRLT','DV','DVC','DVPA','DVPDP','DVPSP_F','EBIT','EBITDA','EMOL','EMP','EPSFI','EPSPI','ESOPCT','ESOPDLT','EXCHG','EXRE','FATB','FATE','FATN','FATL','FATO','FCA','FEL','FIC','FINCF','FOPT','FYEAR','FYR','GDWL','GDWLAM','GDWLIA','GLA','GP','GWO','HEDGEGL','ICAPT','IDBFLAG','IDIIS','IDILC','IDIT','INTAN','INTC','INVCH','INCFG','INVOFS','INVRM','INVT','INVWIP','IPODATE','ISEQ','ISEQC','ISEQM','ISFI','ISGR','ISGU','ITCB','IVCH','IVNCF','LCT','LCUACU','LIFR','LLRCI','LLWOCI','LT','MIB','MIBT','MII','MKVALT','NAICS','NI','NIINT','NIINTPFC','NIINTPFP','NOPIO','NP','NPANL','NPAT','NRTXT','OANCF','OB','OIADP','OIBDP','OPEPS','OPINI','OPITI','OPREPSX','OPTCA','OPTEX','OPTEXD','OPTFVGR','OPTGR','OPTPRCCA','OPTPRCEX','OPTPRCEY','OPTPRCGR','OPTPRCWA','OPTVOL','PCL','PI','PIFO','PLL','PNCA','PNCIA','PPEGT','PPENB','PPENC','PPENLS','PPENNR','PPENT','PPEVO','PPEVR','PRCC_F','PRCH_F','PRICAN','PRIROW','PRIUSA','PRODV','PRSTKCC','PRSTKPC','PSTKC','PVCL','PVO','PVPL','PVT','RANK','RCA','RDIP','RE','REA','REAJO','RECCH','RECT','RECTA','REUNA','REVT','RLL','RMUM','RPAG','RVLRV','RVTI','SALE','SALEPFC','SALEPFP','SCO','SCSTKC','SEQ','SIC','SIV','SPCE','SPCINDCD','SPCSECCD','SPCSRC','SPI','SPPE','SPPIV','SPSTKC','SRET','SSNP','STBO','STIO','STKCO','STKO','TDC','TEQ','TFVA','TFVCE','TFVL','TIC','TLCF','TRANSA','TSA','TSTKC','TSTKP','TXC','TXDB','TXDBA','TXDBCA','TXDBCL','TXFED','TXFO','TXPD','TXR','TXT','TXTUBXINTBS','TXTUBXINTIS','VPAC','VPO','WCAP','WCAPC','WCAPCH','WDA','XACC','XAD','XAGO','XAGT','XCOM','XDP','XEQO','XI','XIDO','XINST','XINT','XINTD','XINTOPT','XLR','XOPR','XOPRAR','XPP','XPR','XRD','XRENT','XS','XSGA','XSTF','XSTFO','XSTFWS','XT']
ls = [x.lower() for x in ls]
annual_filtered = annual_compustat.select([c for c in annual_compustat.columns if c in ls])

cond = [dgls['YEARA'] == annual_filtered['fyear'], dgls['COMPUSTAT IDENTIFIER'] == annual_filtered['conm']]

annual_joined = annual_filtered.join(dgls, cond,'left')
final_joined = annual_joined.join(ibes_complete, ibes_complete['OFTIC'] == annual_joined['tic'], 'left')

def createFlag(yeara):
	if yeara is None:
		return 0
	else:
		return 1

createLabelFlag = udf(createFlag, IntegerType())
# Creating a flag field which is equal to 1.0 when there is matching record in DGLS otherwise 0.0
final_joined = final_joined.withColumn('label', createLabelFlag(final_joined.YEARA))

ls.append('label')
final_joined = final_joined.drop('FYR').drop('COGS')
final_to_keep = final_joined.select([c for c in final_joined.columns if c in ls])

nullcounts = final_to_keep.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in final_to_keep.columns])
nc = list(nullcounts.first())

services_prepacked_software = final_to_keep.filter(final_to_keep.sic == '7372')

some_dict = {}
for x in services_prepacked_software.columns:
	some_dict[x] = 0

nwdf = services_prepacked_software.fillna(some_dict)

good_columns = []
for i in range(0, len(nc)):
	if nc[i] == 0:
		good_columns.append(i)

great_columns = [nwdf.columns[i] for i in good_columns]
nwdf = nwdf.fillna(some_dict)

non_string_columns = [k for (k,v) in nwdf.dtypes if v != 'string']
nwdf_no_strings = nwdf.select(*non_string_columns)
feature_columns = [item for item in nwdf_no_strings.columns if item not in ['label', 'features']]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(nwdf_no_strings)
final_final_df = final_df.drop(*feature_columns)

clustering_input = final_final_df.select(final_final_df.features)
kmeans = clustering.KMeans(k=2)
clst_model = kmeans.fit(clustering_input)
transformed = clst_model.transform(final_final_df)

transformed.write.parquet('/user/vcs/clustering_output_parquet')
