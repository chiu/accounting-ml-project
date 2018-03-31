from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName('aaerLabelCreation').getOrCreate()

annual_compustat = spark.read.csv('/user/vcs/annual_compustat.csv', header=True, inferSchema=True)
dgls = spark.read.csv('/user/vcs/dgls_integrated.csv', header=True, inferSchema=True)
ibes_complete = spark.read.csv('/user/vcs/ibes.csv', header=True, inferSchema=True)

ls = ['CONM', 'ACCHG', 'ACO', 'ACOMINC', 'ACQCSHI', 'ACQGDWL', 'ACQINTAN', 'ACQLNTAL', 'ACT', 'AMGW','AQ','AP','AQC','ARC','AT','AU','AUOP','AUOPIC','BKVLPS','CAPX','CDVC','CEQ','CGA','CHE','CHECH','CI','CIMII','CLD2','CLD3','CLD4','CLD5','CLT','COGS','CSHI','CURNCD','DCOM','DCPSTK','DCVT','DEPC','DLTT','DM','DO','DP','DPACLS','DRC','DRLT','DV','DVC','DVPA','DVPDP','DVPSP_F','EBIT','EBITDA','EMOL','EMP','EPSFI','EPSPI','ESOPCT','ESOPDLT','EXCHG','EXRE','FATB','FATE','FATN','FATL','FATO','FCA','FEL','FIC','FINCF','FOPT','FYEAR','FYR','GDWL','GDWLAM','GDWLIA','GLA','GP','GWO','HEDGEGL','ICAPT','IDBFLAG','IDIIS','IDILC','IDIT','INTAN','INTC','INVCH','INCFG','INVOFS','INVRM','INVT','INVWIP','IPODATE','ISEQ','ISEQC','ISEQM','ISFI','ISGR','ISGU','ITCB','IVCH','IVNCF','LCT','LCUACU','LIFR','LLRCI','LLWOCI','LT','MIB','MIBT','MII','MKVALT','NAICS','NI','NIINT','NIINTPFC','NIINTPFP','NOPIO','NP','NPANL','NPAT','NRTXT','OANCF','OB','OIADP','OIBDP','OPEPS','OPINI','OPITI','OPREPSX','OPTCA','OPTEX','OPTEXD','OPTFVGR','OPTGR','OPTPRCCA','OPTPRCEX','OPTPRCEY','OPTPRCGR','OPTPRCWA','OPTVOL','PCL','PI','PIFO','PLL','PNCA','PNCIA','PPEGT','PPENB','PPENC','PPENLS','PPENNR','PPENT','PPEVO','PPEVR','PRCC_F','PRCH_F','PRICAN','PRIROW','PRIUSA','PRODV','PRSTKCC','PRSTKPC','PSTKC','PVCL','PVO','PVPL','PVT','RANK','RCA','RDIP','RE','REA','REAJO','RECCH','RECT','RECTA','REUNA','REVT','RLL','RMUM','RPAG','RVLRV','RVTI','SALE','SALEPFC','SALEPFP','SCO','SCSTKC','SEQ','SIC','SIV','SPCE','SPCINDCD','SPCSECCD','SPCSRC','SPI','SPPE','SPPIV','SPSTKC','SRET','SSNP','STBO','STIO','STKCO','STKO','TDC','TEQ','TFVA','TFVCE','TFVL','TIC','TLCF','TRANSA','TSA','TSTKC','TSTKP','TXC','TXDB','TXDBA','TXDBCA','TXDBCL','TXFED','TXFO','TXPD','TXR','TXT','TXTUBXINTBS','TXTUBXINTIS','VPAC','VPO','WCAP','WCAPC','WCAPCH','WDA','XACC','XAD','XAGO','XAGT','XCOM','XDP','XEQO','XI','XIDO','XINST','XINT','XINTD','XINTOPT','XLR','XOPR','XOPRAR','XPP','XPR','XRD','XRENT','XS','XSGA','XSTF','XSTFO','XSTFWS','XT']

ls = [x.lower() for x in ls]

annual_filtered = annual_compustat.select([c for c in annual_compustat.columns if c in ls])

cond = [dgls['YEARA'] == annual_filtered['fyear'], dgls['COMPUSTAT IDENTIFIER'] == annual_filtered['conm']]

annual_joined = annual_filtered.join(dgls, cond,'left')
final_joined = annual_joined.join(ibes_complete, ibes_complete['OFTIC'] == annual_joined['tic'], 'left')

def createFlag(yeara):
	if yeara is None:
		return 0.0
	else:
		return 1.0

createLabelFlag = udf(createFlag, DoubleType())

# Creating a flag field which is equal to 1.0 when there is matching record in DGLS otherwise 0.0
final_joined = final_joined.withColumn('label', createLabelFlag(final_joined.YEARA))

ls.append('label')
final_joined = final_joined.drop('FYR').drop('COGS')
final_to_keep = final_joined.select([c for c in final_joined.columns if c in ls])


# Writing the joinned csv / parquet to hdfs
try:
    final_to_keep.coalesce(1).write.csv('/user/vcs/integrated_dataset_with_labels')
except:
    pass

try:
    final_to_keep.write.parquet('annual_integrated_dataset_with_labels.parquet')
except:
    pass
