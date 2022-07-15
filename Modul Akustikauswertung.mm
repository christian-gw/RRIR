<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1657609147811" ID="ID_210302330" MODIFIED="1657609163258" TEXT="Modul Akustikauswertung">
<node CREATED="1657609163982" ID="ID_826095421" MODIFIED="1657609165846" POSITION="right" TEXT="Signal">
<node CREATED="1657609611620" ID="ID_1050644046" MODIFIED="1657609620812" TEXT="Signal">
<node CREATED="1657609620812" ID="ID_1999891158" MODIFIED="1657609636694" TEXT="__init__">
<node CREATED="1657609650639" ID="ID_1157769074" MODIFIED="1657609807830" TEXT="path=str, name=str"/>
<node CREATED="1657609663666" ID="ID_103113640" MODIFIED="1657609856796" TEXT="sweep_par=(start_freq=float, end_freq=float, T=float), dt=float"/>
<node CREATED="1657609860721" ID="ID_1506466572" MODIFIED="1657609889702" TEXT="singal_lst_imp=[Signal, Signal, ...]"/>
<node CREATED="1657609890578" ID="ID_632132835" MODIFIED="1657609910831" TEXT="y=np.array, dt=np.float"/>
</node>
<node CREATED="1657609975728" ID="ID_92134231" MODIFIED="1657609997256" TEXT="__load_data(self, path=str)"/>
<node CREATED="1657610002162" ID="ID_176481359" MODIFIED="1657610032008" TEXT="__gen_sweep(sweep_par=(start_freq=float, end_freq=float, T=float), dt=float)"/>
<node CREATED="1657610033450" ID="ID_101244222" MODIFIED="1657610049823" TEXT="__inv_u(self, x=Signal)"/>
<node CREATED="1657610055986" ID="ID_463403025" MODIFIED="1657610063203" TEXT="__fft_all(self)"/>
<node CREATED="1657609911600" ID="ID_698849973" MODIFIED="1657610082624" TEXT="impulse_response(self, exitaion=Signal)"/>
<node CREATED="1657609925051" ID="ID_953479131" MODIFIED="1657610113711" TEXT="filter_y(self, frange=(float, float))"/>
<node CREATED="1657609928952" ID="ID_521192905" MODIFIED="1657610127269" TEXT="resample(self, Fs_new=float)"/>
<node CREATED="1657609931281" ID="ID_1591149169" MODIFIED="1657610173676" TEXT="cut_signal(self, t_start=float, t_end=float, force_n=int"/>
<node CREATED="1657609936990" ID="ID_110494393" MODIFIED="1657610189106" TEXT="plot_y_t(self, headline=str)"/>
<node CREATED="1657609941588" ID="ID_1495492167" MODIFIED="1657610194304" TEXT="plot_y_f(self, headline=str)"/>
<node CREATED="1657609949313" ID="ID_1864284203" MODIFIED="1657610214794" TEXT="plot_spec_transform(self, n_win=int)"/>
<node CREATED="1657610215807" ID="ID_842541917" MODIFIED="1657610243133" TEXT="level_time(self, T=float)"/>
<node CREATED="1657610243673" ID="ID_1160275907" MODIFIED="1657610268315" TEXT="write_waf(self, name=str, F_samp=int)"/>
<node CREATED="1657610269075" ID="ID_449946972" MODIFIED="1657610334524" TEXT="correct_refl_component(self, direct=Signal, t_start=float, t_dur=float)"/>
</node>
<node CREATED="1657610344187" ID="ID_46783528" MODIFIED="1657610403736" TEXT="rotate_sig_lst(sig_lst=[Signal, Signal, ...], cor_range_pre=int, start_cor_reference=int, start_cor_reflection=int, fix_shift=int)"/>
<node CREATED="1657610412985" ID="ID_9280710" MODIFIED="1657610473516" TEXT="to_db(x=float/np.array(float))"/>
<node CREATED="1657610478260" ID="ID_765753245" MODIFIED="1657610541061" TEXT="appl_win(sig=Signal, t_start,=float, t_len=float, form=str)"/>
<node CREATED="1657610547395" ID="ID_930291994" MODIFIED="1657610571563" TEXT="create_band_lst(fact=float)"/>
<node CREATED="1657610572668" ID="ID_641981037" MODIFIED="1657610628678" TEXT="pre_process_one_measurement(path=str, sig_name=str, F_up=float, u=Signal)"/>
</node>
<node CREATED="1657609168837" ID="ID_1283125918" MODIFIED="1657609203695" POSITION="right" TEXT="Transfer_Function">
<node CREATED="1657609334897" ID="ID_1402608013" MODIFIED="1657609342808" TEXT="TransferFunction">
<node CREATED="1657609342808" ID="ID_1547701209" MODIFIED="1657609352841" TEXT="__init__">
<node CREATED="1657609352841" ID="ID_727416025" MODIFIED="1657609693318" TEXT="incoming_sig=Signal, reflected_sig=Signal"/>
<node CREATED="1657609379947" ID="ID_1334672145" MODIFIED="1657609709138" TEXT="xf=np.array, hf=np.array"/>
<node CREATED="1657609393245" ID="ID_1323762740" MODIFIED="1657609746178" TEXT="signal=Signal, in_win=[float, float], re_win=[float, float]"/>
</node>
<node CREATED="1657609457580" ID="ID_1437557428" MODIFIED="1657609499145" TEXT="plot_hf(self)"/>
<node CREATED="1657609462487" ID="ID_1714446546" MODIFIED="1657609756446" TEXT="convolute(self, sig=Signal)"/>
<node CREATED="1657609501116" ID="ID_1737498082" MODIFIED="1657609771132" TEXT="__get_band(self, f0=float, f1=float)"/>
<node CREATED="1657609519220" ID="ID_561664529" MODIFIED="1657609776934" TEXT="get_octave_band(self, fact=float)"/>
</node>
</node>
<node CREATED="1657609180987" ID="ID_773605540" MODIFIED="1657609183771" POSITION="right" TEXT="Reflection">
<node CREATED="1657612392000" ID="ID_464583689" MODIFIED="1657612394791" TEXT="Measurement">
<node CREATED="1657612406872" ID="ID_205264534" MODIFIED="1657612872630" TEXT="__init__(self, name=str, d_mic=float, d_probe=float)"/>
<node CREATED="1657612419608" ID="ID_1075109202" MODIFIED="1657612845427" TEXT="create_mp(self, number=int, _signal=Signal, pos[x=float, y=float])"/>
<node CREATED="1657612425198" ID="ID_1729197698" MODIFIED="1657612792210" TEXT="del_mp(self, number=int)"/>
<node CREATED="1657612430594" ID="ID_273205207" MODIFIED="1657612775031" TEXT="plot_overview(self)"/>
<node CREATED="1657612438292" ID="ID_236118576" MODIFIED="1657612767626" TEXT="average_mp(self)"/>
</node>
<node CREATED="1657612395398" ID="ID_1417219820" MODIFIED="1657612403790" TEXT="Measurement_Point">
<node CREATED="1657612447812" ID="ID_1397056984" MODIFIED="1657612728811" TEXT="__init__(self, number=int, distances=[src-mic=float, mic-probe=float], transfer_function=Transfer_function, pos=[x=float, y=float])"/>
<node CREATED="1657612459371" ID="ID_255534707" MODIFIED="1657612564773" TEXT="__geo_norm(self)"/>
<node CREATED="1657612466736" ID="ID_922008131" MODIFIED="1657612556056" TEXT="calc_c_geo(self, norm=bool"/>
<node CREATED="1657612473499" ID="ID_1274165389" MODIFIED="1657612544373" TEXT="apply_c(self)"/>
<node CREATED="1657612479520" ID="ID_754494766" MODIFIED="1657612519132" TEXT="calc_c_dir(self)"/>
<node CREATED="1657612486711" ID="ID_1283918172" MODIFIED="1657612510414" TEXT="beta_in_deg(self)"/>
</node>
</node>
<node CREATED="1657609176994" ID="ID_783781403" MODIFIED="1657609179224" POSITION="right" TEXT="Ambi">
<node CREATED="1657613328801" ID="ID_1578059634" MODIFIED="1657613333056" TEXT="Mic">
<node CREATED="1657613365161" ID="ID_1275870357" MODIFIED="1657615541998" TEXT="__init__(self, alpha: float)"/>
<node CREATED="1657613333056" ID="ID_1077510824" MODIFIED="1657615582375" TEXT="__basic_pattern_2d(self, phi: float/np.array(float))"/>
<node CREATED="1657613347940" ID="ID_32920955" MODIFIED="1657615637784" TEXT="__basic_pattern_3d(self, phi: float/np.array(float), theta float/np.array(float))"/>
<node CREATED="1657613370781" ID="ID_1990116529" MODIFIED="1657615600845" TEXT="plot_directivity_2d(self)"/>
<node CREATED="1657613380260" ID="ID_1606589321" MODIFIED="1657615650101" TEXT="plot_directivity_3d(self)"/>
</node>
<node CREATED="1657613398927" ID="ID_61752126" MODIFIED="1657613410141" TEXT="AmbiMic">
<node CREATED="1657613410141" ID="ID_144147776" MODIFIED="1657614516813" TEXT="__init__(self, R=float, a=float, c=float)"/>
<node CREATED="1657613414326" ID="ID_1023445769" MODIFIED="1657613493675" TEXT="_calc_M_AB(self)"/>
<node CREATED="1657613424555" ID="ID_1749617550" MODIFIED="1657613482129" TEXT="_calc_positions(self)"/>
<node CREATED="1657613439366" ID="ID_498853607" MODIFIED="1657613480307" TEXT="_calc_hw_cor_Gerzon(self)"/>
<node CREATED="1657613446207" ID="ID_1267805661" MODIFIED="1657613474280" TEXT="_calc_hxyz_cor_Gerzon(self)"/>
</node>
<node CREATED="1657614529613" ID="ID_498719112" MODIFIED="1657614536291" TEXT="ambiSig">
<node CREATED="1657614541894" ID="ID_1323857457" MODIFIED="1657614590605" TEXT="__init__(self, Signals: list, mic_settings: AmbiMic)"/>
<node CREATED="1657614591246" ID="ID_34982073" MODIFIED="1657614608367" TEXT="__create_b_format(self, mic: AmbiMic)"/>
<node CREATED="1657614615151" ID="ID_1329596481" MODIFIED="1657614645380" TEXT="__create_rot_matrix(self, phi: float, theta: float, rad: Bool)"/>
<node CREATED="1657614646363" ID="ID_1037772210" MODIFIED="1657614686948" TEXT="_rotate_b_format(self, angle: np.array(3,3))"/>
<node CREATED="1657614687619" ID="ID_1008731643" MODIFIED="1657614718144" TEXT="_extract_dir_signal(self, angle:np.array: np.array(3,3))"/>
<node CREATED="1657614719888" ID="ID_917622864" MODIFIED="1657614734768" TEXT="safe_v_format(self, names: dict)"/>
</node>
</node>
</node>
</map>
