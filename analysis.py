import pandas as pd
import matplotlib.pyplot as plt


def load_dtc_data():
    with open('dtc_results.txt') as f:
        dt_train = list()
        dt_test = list()
        dt_depth = list()
        dt_features = list()
        dt_cw = list()

        for line in f:
            if line.startswith('#'):
                continue
            rec = line.strip().split(', ')

            dt_train.append(float(rec[0]))
            dt_test.append(float(rec[1]))
            dt_depth.append(int(rec[2]))
            dt_features.append(rec[3])
            dt_cw.append(rec[4])

    dtdf = pd.DataFrame([dt_train, dt_test, dt_depth, dt_features, dt_cw]).T
    return dtdf


def load_rfc_data():
    with open('rfc_results.txt') as f:
        rf_train = list()
        rf_test = list()
        rf_depth = list()
        rf_features = list()
        rf_cw = list()

        for line in f:
            if line.startswith('#'):
                continue
            rec = line.strip().split(', ')

            rf_train.append(float(rec[0]))
            rf_test.append(float(rec[1]))
            rf_depth.append(int(rec[2]))
            rf_features.append(rec[3])
            rf_cw.append(rec[4])

    rfdf = pd.DataFrame([rf_train, rf_test, rf_depth, rf_features, rf_cw]).T
    return rfdf


def generate_all_visuals(dtdf, rfdf):
    # prep DTC DFs
    dtdf_weighted = dtdf.loc[dtdf[4] == 'dictionary']
    dtdf_None = dtdf.loc[dtdf[4] == 'None']

    dtdf_weighted_mfNone = dtdf_weighted.loc[dtdf[3] == 'None']
    dtdf_weighted_mfsqrt = dtdf_weighted.loc[dtdf[3] == 'sqrt']
    dtdf_None_mfNone = dtdf_None.loc[dtdf[3] == 'None']
    dtdf_None_mfsqrt = dtdf_None.loc[dtdf[3] == 'sqrt']

    # prep RFC DFs
    rfdf_weighted = rfdf.loc[rfdf[4] == 'dictionary']
    rfdf_None = rfdf.loc[rfdf[4] == 'None']

    rfdf_weighted_mfNone = rfdf_weighted.loc[rfdf[3] == 'None']
    rfdf_weighted_mfsqrt = rfdf_weighted.loc[rfdf[3] == 'sqrt']
    rfdf_None_mfNone = rfdf_None.loc[rfdf[3] == 'None']
    rfdf_None_mfsqrt = rfdf_None.loc[rfdf[3] == 'sqrt']

    #---------- DTC_Weighted_MaxFeatures ----------

    plt.plot(dtdf_weighted_mfNone[2], dtdf_weighted_mfNone[1], label='None/all')
    plt.plot(dtdf_weighted_mfsqrt[2], dtdf_weighted_mfsqrt[1], label='sqrt')
    plt.title('(With class weights) DTC Max Features: All vs Sqrt')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/DTC_Weighted_MaxFeatures.png')
    plt.clf()

    # ---------- DTC_NoWeight_MaxFeatures ----------
    plt.plot(dtdf_None_mfNone[2], dtdf_None_mfNone[1], label='None/all')
    plt.plot(dtdf_None_mfsqrt[2], dtdf_None_mfsqrt[1], label='sqrt')
    plt.title('(No class weights) DTC Max Features: All vs Sqrt')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/DTC_NoWeight_MaxFeatures.png')
    plt.clf()

    # ---------- RFC_Weighted_MaxFeatures ----------
    plt.plot(rfdf_weighted_mfNone[2], rfdf_weighted_mfNone[1], label='None/all')
    plt.plot(rfdf_weighted_mfsqrt[2], rfdf_weighted_mfsqrt[1], label='sqrt')
    plt.title('(With class weights) RFC Max Features: All vs Sqrt')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/RFC_Weighted_MaxFeatures.png')
    plt.clf()

    # ---------- RFC_NoWeight_MaxFeatures ----------
    plt.plot(rfdf_None_mfNone[2], rfdf_None_mfNone[1], label='None/all')
    plt.plot(rfdf_None_mfsqrt[2], rfdf_None_mfsqrt[1], label='sqrt')
    plt.title('(No class weights) RFC Max Features: All vs Sqrt')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/RFC_NoWeight_MaxFeatures.png')
    plt.clf()

    # ---------- RFC_vs_DTC_NoWeight_SQRT ----------
    plt.plot(rfdf_None_mfsqrt[2], rfdf_None_mfsqrt[1], label='RFC')
    plt.plot(dtdf_None_mfsqrt[2], dtdf_None_mfsqrt[1], label='DTC')
    plt.title('(No class weights, sqrt features) RFC vs DTC')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/RFC_vs_DTC_NoWeight_SQRT.png')
    plt.clf()

    # ---------- RFC_vs_DTC_Weighted_SQRT ----------
    plt.plot(rfdf_weighted_mfsqrt[2], rfdf_weighted_mfsqrt[1], label='RFC')
    plt.plot(dtdf_weighted_mfsqrt[2], dtdf_weighted_mfsqrt[1], label='DTC')
    plt.title('(With class weights, sqrt features) RFC vs DTC')

    plt.legend()
    plt.ylabel('Test accuracy')
    plt.xlabel('Max depth')

    plt.savefig('images/RFC_vs_DTC_Weighted_SQRT.png')
    plt.clf()

    print('\n---------- Generated all visuals (see images folder) ----------')
