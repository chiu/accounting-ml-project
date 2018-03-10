from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

annual_compustat = spark.read.csv('/user/vcs/annual_compustat.csv', header=True)
balance_sheet = spark.read.csv('/user/vcs/balance_sheet.csv', header=True)
income_stat = spark.read.csv('/user/vcs/income_statements.csv', header=True)
offbalance_sheet = spark.read.csv('/user/vcs/offbalance_sheet_items.csv', header=True)
risk_cap = spark.read.csv('/user/vcs/risk_based_capital.csv', header=True)
ibes_predictions = spark.read.csv('/user/vcs/sample_ibes.csv', header=True)
dgls = spark.read.csv('/user/vcs/dgls_integrated.csv', header=True)
ibes_complete = spark.read.csv('/user/vcs/ibes.csv', header=True)

# Number of total records
print(annual_compustat.count())
print(balance_sheet.count())
print(income_stat.count())
print(offbalance_sheet.count())
print(risk_cap.count())
print(ibes_predictions.count())
print(dgls.count())
print(ibes_complete.count())

annual_cols = annual_compustat.schema.names
balance_cols = balance_sheet.schema.names
income_cols = income_stat.schema.names
offbalance_cols = offbalance_sheet.schema.names
risk_cols = risk_cap.schema.names
ibes_cols = ibes_predictions.schema.names
dgls_cols = dgls.schema.names
ibes_complete_cols = ibes_complete.schema.names

ls = ['CONM', 'ACCHG', 'ACO', 'ACOMINC', 'ACQCSHI', 'ACQGDWL', 'ACQINTAN', 'ACQLNTAL', 'ACT', 'AMGW','AQ','AP','AQC','ARC','AT','AU','AUOP','AUOPIC','BKVLPS','CAPX','CDVC','CEQ','CGA','CHE','CHECH','CI','CIMII','CLD2','CLD3','CLD4','CLD5','CLT','COGS','CSHI','CURNCD','DCOM','DCPSTK','DCVT','DEPC','DLTT','DM','DO','DP','DPACLS','DRC','DRLT','DV','DVC','DVPA','DVPDP','DVPSP_F','EBIT','EBITDA','EMOL','EMP','EPSFI','EPSPI','ESOPCT','ESOPDLT','EXCHG','EXRE','FATB','FATE','FATN','FATL','FATO','FCA','FEL','FIC','FINCF','FOPT','FYEAR','FYR','GDWL','GDWLAM','GDWLIA','GLA','GP','GWO','HEDGEGL','ICAPT','IDBFLAG','IDIIS','IDILC','IDIT','INTAN','INTC','INVCH','INCFG','INVOFS','INVRM','INVT','INVWIP','IPODATE','ISEQ','ISEQC','ISEQM','ISFI','ISGR','ISGU','ITCB','IVCH','IVNCF','LCT','LCUACU','LIFR','LLRCI','LLWOCI','LT','MIB','MIBT','MII','MKVALT','NAICS','NI','NIINT','NIINTPFC','NIINTPFP','NOPIO','NP','NPANL','NPAT','NRTXT','OANCF','OB','OIADP','OIBDP','OPEPS','OPINI','OPITI','OPREPSX','OPTCA','OPTEX','OPTEXD','OPTFVGR','OPTGR','OPTPRCCA','OPTPRCEX','OPTPRCEY','OPTPRCGR','OPTPRCWA','OPTVOL','PCL','PI','PIFO','PLL','PNCA','PNCIA','PPEGT','PPENB','PPENC','PPENLS','PPENNR','PPENT','PPEVO','PPEVR','PRCC_F','PRCH_F','PRICAN','PRIROW','PRIUSA','PRODV','PRSTKCC','PRSTKPC','PSTKC','PVCL','PVO','PVPL','PVT','RANK','RCA','RDIP','RE','REA','REAJO','RECCH','RECT','RECTA','REUNA','REVT','RLL','RMUM','RPAG','RVLRV','RVTI','SALE','SALEPFC','SALEPFP','SCO','SCSTKC','SEQ','SIC','SIV','SPCE','SPCINDCD','SPCSECCD','SPCSRC','SPI','SPPE','SPPIV','SPSTKC','SRET','SSNP','STBO','STIO','STKCO','STKO','TDC','TEQ','TFVA','TFVCE','TFVL','TIC','TLCF','TRANSA','TSA','TSTKC','TSTKP','TXC','TXDB','TXDBA','TXDBCA','TXDBCL','TXFED','TXFO','TXPD','TXR','TXT','TXTUBXINTBS','TXTUBXINTIS','VPAC','VPO','WCAP','WCAPC','WCAPCH','WDA','XACC','XAD','XAGO','XAGT','XCOM','XDP','XEQO','XI','XIDO','XINST','XINT','XINTD','XINTOPT','XLR','XOPR','XOPRAR','XPP','XPR','XRD','XRENT','XS','XSGA','XSTF','XSTFO','XSTFWS','XT']

ls = [x.lower() for x in ls]

annual_filtered = annual_compustat.select([c for c in annual_compustat.columns if c in ls])

annual_joined = annual_filtered.join(dgls, dgls['COMPUSTAT IDENTIFIER'] == annual_filtered['conm'],'left')
final_joined = annual_joined.join(ibes_complete, ibes_complete['OFTIC'] == annual_joined['tic'], 'left')

print(final_joined.show())

# Renaming column names wit witespaces (parquet doesnt allow spaces as colnames)
final_joined = final_joined.withColumnRenamed('COMPUSTAT IDENTIFIER', 'compustat_identifier').withColumnRenamed('AAER DATABASE IDENTIFIER', 'aaer_db_identifier')

drop_list = ['COMPUSTAT IDENTIFIER', 'AAER DATABASE IDENTIFIER']

final_joined = final_joined.drop(dgls.FYR)
# final_joined = final_joined.select([column for column in list(set(final_joined.columns)) if column not in drop_list])
#
#
# print(final_joined.show())
#
# print(final_joined.schema.names)
#
# print(set(final_joined.schema.names))
#
# # Writing the joinned csv to hdfs
# final_joined.coalesce(1).write.csv('integrated_dataset')
# final_joined.write.parquet('/user/vcs/annual_integrated_dataset_parquet')
