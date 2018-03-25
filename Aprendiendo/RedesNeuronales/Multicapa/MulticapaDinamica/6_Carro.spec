# -*- mode: python -*-

block_cipher = None


a = Analysis(['6_Carro.py'],
             pathex=['/home/louiso/Escritorio/CienciaDeLaComputacion/Ciclo_2017_1/IntelifenciaArtificial/Aprendiendo/RedesNeuronales/Multicapa/MulticapaDinamica'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='6_Carro',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
