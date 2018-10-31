import csv
import mir_eval
import numpy as np
import pickle

import pandas as pd
def getlist_mdb():
    train = ["AimeeNorwich_Child", "AimeeNorwich_Flying", "AlexanderRoss_GoodbyeBolero", "AlexanderRoss_VelvetCurtain", "AvaLuna_Waterduct", "BigTroubles_Phantom", "CroqueMadame_Oil", "CroqueMadame_Pilot", "DreamersOfTheGhetto_HeavyLove", "EthanHein_1930sSynthAndUprightBass", "EthanHein_GirlOnABridge", "FacesOnFilm_WaitingForGa", "FamilyBand_Again", "Handel_TornamiAVagheggiar", "HeladoNegro_MitadDelMundo", "HopAlong_SisterCities", "JoelHelander_Definition", "JoelHelander_ExcessiveResistancetoChange", "JoelHelander_IntheAtticBedroom", "KarimDouaidy_Hopscotch", "KarimDouaidy_Yatora", "LizNelson_Coldwar", "LizNelson_ImComingHome", "LizNelson_Rainfall", "Meaxic_TakeAStep", "Meaxic_YouListen", "Mozart_BesterJungling", "MusicDelta_80sRock", "MusicDelta_Beatles", "MusicDelta_BebopJazz", "MusicDelta_Beethoven", "MusicDelta_Britpop", "MusicDelta_ChineseChaoZhou", "MusicDelta_ChineseDrama", "MusicDelta_ChineseHenan", "MusicDelta_ChineseJiangNan", "MusicDelta_ChineseXinJing", "MusicDelta_ChineseYaoZu", "MusicDelta_CoolJazz", "MusicDelta_Country1", "MusicDelta_Country2", "MusicDelta_Disco", "MusicDelta_FreeJazz", "MusicDelta_FunkJazz", "MusicDelta_GriegTrolltog", "MusicDelta_Grunge", "MusicDelta_Hendrix", "MusicDelta_InTheHalloftheMountainKing", "MusicDelta_LatinJazz", "MusicDelta_ModalJazz", "MusicDelta_Punk", "MusicDelta_Reggae", "MusicDelta_Rock", "MusicDelta_Rockabilly", "MusicDelta_Shadows", "MusicDelta_SpeedMetal", "MusicDelta_Vivaldi", "MusicDelta_Zeppelin", "PurlingHiss_Lolita", "Schumann_Mignon", "StevenClark_Bounty", "SweetLights_YouLetMeDown", "TheDistricts_Vermont", "TheScarletBrand_LesFleursDuMal", "TheSoSoGlos_Emergency", "Wolf_DieBekherte"]
    validation = ["AmarLal_Rest", "AmarLal_SpringDay1", "BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly", "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella", "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims", "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "MichaelKropf_AllGoodThings", "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]
    test = ["AClassicEducation_NightOwl", "Auctioneer_OurFutureFaces", "CelestialShore_DieForUs", "ChrisJacoby_BoothShotLincoln", "ChrisJacoby_PigsFoot", "Creepoid_OldTree", "Debussy_LenfantProdigue", "MatthewEntwistle_DontYouEver", "MatthewEntwistle_FairerHopes", "MatthewEntwistle_ImpressionsOfSaturn", "MatthewEntwistle_Lontano", "MatthewEntwistle_TheArch", "MatthewEntwistle_TheFlaxenField", "Mozart_DiesBildnis", "MusicDelta_FusionJazz", "MusicDelta_Gospel", "MusicDelta_Pachelbel", "MusicDelta_SwingJazz", "Phoenix_BrokenPledgeChicagoReel", "Phoenix_ColliersDaughter", "Phoenix_ElzicsFarewell", "Phoenix_LarkOnTheStrandDrummondCastle", "Phoenix_ScotchMorris", "Phoenix_SeanCaughlinsTheScartaglen", "PortStWillow_StayEven", "Schubert_Erstarrung", "StrandOfOaks_Spacestation"]
    return train, validation, test

def getlist_mdb_vocal():
    train_songlist = ["AimeeNorwich_Child", "AlexanderRoss_GoodbyeBolero", "AlexanderRoss_VelvetCurtain", "AvaLuna_Waterduct", "BigTroubles_Phantom", "DreamersOfTheGhetto_HeavyLove", "FacesOnFilm_WaitingForGa", "FamilyBand_Again", "Handel_TornamiAVagheggiar", "HeladoNegro_MitadDelMundo", "HopAlong_SisterCities", "LizNelson_Coldwar", "LizNelson_ImComingHome", "LizNelson_Rainfall", "Meaxic_TakeAStep", "Meaxic_YouListen", "MusicDelta_80sRock", "MusicDelta_Beatles", "MusicDelta_Britpop", "MusicDelta_Country1", "MusicDelta_Country2", "MusicDelta_Disco", "MusicDelta_Grunge", "MusicDelta_Hendrix", "MusicDelta_Punk", "MusicDelta_Reggae", "MusicDelta_Rock", "MusicDelta_Rockabilly", "PurlingHiss_Lolita", "StevenClark_Bounty", "SweetLights_YouLetMeDown", "TheDistricts_Vermont", "TheScarletBrand_LesFleursDuMal", "TheSoSoGlos_Emergency", "Wolf_DieBekherte"]
    val_songlist = ["BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly", "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella", "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims", "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "Mozart_DiesBildnis", "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]
    test_songlist = ["AClassicEducation_NightOwl", "Auctioneer_OurFutureFaces", "CelestialShore_DieForUs", "Creepoid_OldTree", "Debussy_LenfantProdigue", "MatthewEntwistle_DontYouEver", "MatthewEntwistle_Lontano", "Mozart_BesterJungling", "MusicDelta_Gospel", "PortStWillow_StayEven", "Schubert_Erstarrung", "StrandOfOaks_Spacestation"]
    return train_songlist, val_songlist, test_songlist

def getlist_ADC2004():
    test_songlist = ["daisy1", "daisy2", "daisy3", "daisy4", "opera_fem2", "opera_fem4", "opera_male3", "opera_male5", "pop1", "pop2", "pop3", "pop4"]
    return test_songlist

def getlist_MIREX05():
    test_songlist = ["train01", "train02", "train03", "train04", "train05", "train06", "train07", "train08", "train09"]
    return test_songlist
def melody_eval(ref, est):

    ref_time = ref[:,0]
    ref_freq = ref[:,1]

    est_time = est[:,0]
    est_freq = est[:,1]

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr
def csv2ref(ypath):
    ycsv = pd.read_csv(ypath, names = ["time", "freq"])
    gtt = ycsv['time'].values
    gtf = ycsv['freq'].values
    ref_arr = np.concatenate((gtt[:,None], gtf[:,None]), axis=1)
    return ref_arr

def select_vocal_track(ypath, lpath):
    
    ycsv = pd.read_csv(ypath, names = ["time", "freq"])
    gt0 = ycsv['time'].values
    gt0 = gt0[:,np.newaxis]

    gt1 = ycsv['freq'].values
    gt1 = gt1[:,np.newaxis]

    z = np.zeros(gt1.shape)

    f=open(lpath,'r')
    lines=f.readlines()

    for line in lines:

        if 'start_time' in line.split(',')[0]:
            continue
        st = float(line.split(',')[0])
        et = float(line.split(',')[1])
        sid = line.split(',')[2]
        for i in range(len(gt1)):
            if st < gt0[i,0] < et and 'singer' in sid:
                z[i,0] = gt1[i,0]

    gt = np.concatenate((gt0,z),axis=1)
    return gt

def save_csv(data, savepath):
    with open(savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['VR', 'VFA', 'RPA', 'RCA', 'OA'])
        for est_arr in data:
            writer.writerow(est_arr)
def load_list(savepath):
    with open(savepath, 'rb') as file:
        xlist = pickle.load(file)
    return xlist

